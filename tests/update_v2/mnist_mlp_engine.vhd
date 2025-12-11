library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mnist_mlp_pkg.all;

entity mnist_mlp_engine is
    port (
        clk        : in  std_logic;
        rst_n      : in  std_logic;
        start_i    : in  std_logic;
        busy_o     : out std_logic;
        done_o     : out std_logic;
        error_o    : out std_logic;

        -- Image BRAM interface (Port B)
        img_addr_o : out unsigned(15 downto 0);
        img_data_i : in  std_logic_vector(31 downto 0);

        -- Final logits (INT8)
        logits_out : out int8_vector_t(0 to L4_OUT_SIZE-1)
    );
end entity mnist_mlp_engine;

architecture rtl of mnist_mlp_engine is

    attribute use_dsp : string;
    attribute use_dsp of rtl : architecture is "yes";

    -- ROM entities (weights / bias)
    component weight_rom is
        generic (
            INIT_FILE  : string;
            WORD_COUNT : integer;
            ADDR_WIDTH : integer
        );
        port (
            clk  : in  std_logic;
            addr : in  unsigned(ADDR_WIDTH-1 downto 0);
            dout : out weight_word_t
        );
    end component;

    component bias_rom is
        generic (
            INIT_FILE  : string;
            WORD_COUNT : integer;
            ADDR_WIDTH : integer
        );
        port (
            clk  : in  std_logic;
            addr : in  unsigned(ADDR_WIDTH-1 downto 0);
            dout : out int32_t
        );
    end component;

    -- ROM interface signals
    signal w1_addr : unsigned(W1_ADDR_WIDTH-1 downto 0);
    signal w1_dout : weight_word_t;
    signal b1_addr : unsigned(B1_ADDR_WIDTH-1 downto 0);
    signal b1_dout : int32_t;

    signal w2_addr : unsigned(W2_ADDR_WIDTH-1 downto 0);
    signal w2_dout : weight_word_t;
    signal b2_addr : unsigned(B2_ADDR_WIDTH-1 downto 0);
    signal b2_dout : int32_t;

    signal w3_addr : unsigned(W3_ADDR_WIDTH-1 downto 0);
    signal w3_dout : weight_word_t;
    signal b3_addr : unsigned(B3_ADDR_WIDTH-1 downto 0);
    signal b3_dout : int32_t;

    signal w4_addr : unsigned(W4_ADDR_WIDTH-1 downto 0);
    signal w4_dout : weight_word_t;
    signal b4_addr : unsigned(B4_ADDR_WIDTH-1 downto 0);
    signal b4_dout : int32_t;

    -- Activation buffers
    signal act_l1 : int8_vector_t(0 to L1_OUT_SIZE-1);
    signal act_l2 : int8_vector_t(0 to L2_OUT_SIZE-1);
    signal act_l3 : int8_vector_t(0 to L3_OUT_SIZE-1);
    signal act_l4 : int8_vector_t(0 to L4_OUT_SIZE-1);

    -- Temporary buffer for L1 (due to 32-bit image BRAM)
    signal img_buf_low : int8_vector_t(0 to 3);

    -- Parallel MAC products (N_PAR)
    signal pipe_prods : int32_vector_t(0 to N_PAR-1) := (others => (others => '0'));

    -- FSM
    type state_t is (
        S_IDLE,

        -- Layer 1
        S_L1_SETUP,
        S_L1_FETCH_A,
        S_L1_FETCH_B,
        S_L1_ACCUM,
        S_L1_QUANT_CALC,
        S_L1_QUANT_APPLY,

        -- Layer 2
        S_L2_SETUP,
        S_L2_BIAS_WAIT,
        S_L2_FETCH,
        S_L2_ACCUM,
        S_L2_QUANT_CALC,
        S_L2_QUANT_APPLY,

        -- Layer 3
        S_L3_SETUP,
        S_L3_BIAS_WAIT,
        S_L3_FETCH,
        S_L3_ACCUM,
        S_L3_QUANT_CALC,
        S_L3_QUANT_APPLY,

        -- Layer 4
        S_L4_SETUP,
        S_L4_BIAS_WAIT,
        S_L4_FETCH,
        S_L4_ACCUM,
        S_L4_QUANT_CALC,
        S_L4_QUANT_APPLY,

        S_DONE,
        S_ERROR
    );
    signal state : state_t := S_IDLE;

    -- Indices and accumulators
    signal l1_neuron_idx : integer range 0 to L1_OUT_SIZE-1 := 0;
    signal l1_block_idx  : integer range 0 to L1_IN_BLOCKS-1 := 0;

    signal l2_neuron_idx : integer range 0 to L2_OUT_SIZE-1 := 0;
    signal l2_block_idx  : integer range 0 to L2_IN_BLOCKS-1 := 0;

    signal l3_neuron_idx : integer range 0 to L3_OUT_SIZE-1 := 0;
    signal l3_block_idx  : integer range 0 to L3_IN_BLOCKS-1 := 0;

    signal l4_neuron_idx : integer range 0 to L4_OUT_SIZE-1 := 0;
    signal l4_block_idx  : integer range 0 to L4_IN_BLOCKS-1 := 0;

    signal acc_reg    : int32_t  := (others => '0');
    signal bias_reg   : int32_t  := (others => '0');
    signal mult_q_reg : signed(63 downto 0) := (others => '0');

    signal busy_reg  : std_logic := '0';
    signal done_reg  : std_logic := '0';
    signal error_reg : std_logic := '0';

begin

    -- ROM instantiations
    u_rom_w1 : weight_rom
        generic map (
            INIT_FILE  => "w1.mem",
            WORD_COUNT => W1_WORD_COUNT,
            ADDR_WIDTH => W1_ADDR_WIDTH
        )
        port map (
            clk  => clk,
            addr => w1_addr,
            dout => w1_dout
        );

    u_rom_w2 : weight_rom
        generic map (
            INIT_FILE  => "w2.mem",
            WORD_COUNT => W2_WORD_COUNT,
            ADDR_WIDTH => W2_ADDR_WIDTH
        )
        port map (
            clk  => clk,
            addr => w2_addr,
            dout => w2_dout
        );

    u_rom_w3 : weight_rom
        generic map (
            INIT_FILE  => "w3.mem",
            WORD_COUNT => W3_WORD_COUNT,
            ADDR_WIDTH => W3_ADDR_WIDTH
        )
        port map (
            clk  => clk,
            addr => w3_addr,
            dout => w3_dout
        );

    u_rom_w4 : weight_rom
        generic map (
            INIT_FILE  => "w4.mem",
            WORD_COUNT => W4_WORD_COUNT,
            ADDR_WIDTH => W4_ADDR_WIDTH
        )
        port map (
            clk  => clk,
            addr => w4_addr,
            dout => w4_dout
        );

    u_rom_b1 : bias_rom
        generic map (
            INIT_FILE  => "b1.mem",
            WORD_COUNT => B1_COUNT,
            ADDR_WIDTH => B1_ADDR_WIDTH
        )
        port map (
            clk  => clk,
            addr => b1_addr,
            dout => b1_dout
        );

    u_rom_b2 : bias_rom
        generic map (
            INIT_FILE  => "b2.mem",
            WORD_COUNT => B2_COUNT,
            ADDR_WIDTH => B2_ADDR_WIDTH
        )
        port map (
            clk  => clk,
            addr => b2_addr,
            dout => b2_dout
        );

    u_rom_b3 : bias_rom
        generic map (
            INIT_FILE  => "b3.mem",
            WORD_COUNT => B3_COUNT,
            ADDR_WIDTH => B3_ADDR_WIDTH
        )
        port map (
            clk  => clk,
            addr => b3_addr,
            dout => b3_dout
        );

    u_rom_b4 : bias_rom
        generic map (
            INIT_FILE  => "b4.mem",
            WORD_COUNT => B4_COUNT,
            ADDR_WIDTH => B4_ADDR_WIDTH
        )
        port map (
            clk  => clk,
            addr => b4_addr,
            dout => b4_dout
        );

    -- Status outputs
    busy_o    <= busy_reg;
    done_o    <= done_reg;
    error_o   <= error_reg;
    logits_out <= act_l4;

    -- Main FSM
    process (clk)
        variable l_inputs : int8_vector_t(0 to N_PAR-1);
        variable v_sum01, v_sum23, v_sum45, v_sum67 : int32_t;
        variable v_sum_total : int32_t;
        variable base_idx : integer;
        variable q64      : signed(63 downto 0);
        variable rounding : signed(63 downto 0);
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                state          <= S_IDLE;
                busy_reg       <= '0';
                done_reg       <= '0';
                error_reg      <= '0';
                img_addr_o     <= (others => '0');
                l1_neuron_idx  <= 0;
                l1_block_idx   <= 0;
                l2_neuron_idx  <= 0;
                l2_block_idx   <= 0;
                l3_neuron_idx  <= 0;
                l3_block_idx   <= 0;
                l4_neuron_idx  <= 0;
                l4_block_idx   <= 0;
                acc_reg        <= (others => '0');
                bias_reg       <= (others => '0');
                mult_q_reg     <= (others => '0');
            else
                done_reg <= '0';

                case state is

                    when S_IDLE =>
                        busy_reg <= '0';
                        if start_i = '1' then
                            busy_reg      <= '1';
                            l1_neuron_idx <= 0;
                            l1_block_idx  <= 0;
                            state         <= S_L1_SETUP;
                        end if;

                    -- LAYER 1
                    when S_L1_SETUP =>
                        b1_addr      <= to_unsigned(l1_neuron_idx, B1_ADDR_WIDTH);
                        w1_addr      <= weight_addr_l1(l1_neuron_idx, 0);
                        l1_block_idx <= 0;
                        acc_reg      <= (others => '0');
                        img_addr_o   <= to_unsigned(0, 16);
                        state        <= S_L1_FETCH_A;

                    when S_L1_FETCH_A =>
                        bias_reg   <= b1_dout;
                        acc_reg    <= b1_dout;
                        img_buf_low(0) <= signed(img_data_i(7  downto 0));
                        img_buf_low(1) <= signed(img_data_i(15 downto 8));
                        img_buf_low(2) <= signed(img_data_i(23 downto 16));
                        img_buf_low(3) <= signed(img_data_i(31 downto 24));
                        img_addr_o     <= to_unsigned(1, 16);
                        state          <= S_L1_FETCH_B;

                    when S_L1_FETCH_B =>
                        l_inputs(0) := img_buf_low(0);
                        l_inputs(1) := img_buf_low(1);
                        l_inputs(2) := img_buf_low(2);
                        l_inputs(3) := img_buf_low(3);
                        l_inputs(4) := signed(img_data_i(7  downto 0));
                        l_inputs(5) := signed(img_data_i(15 downto 8));
                        l_inputs(6) := signed(img_data_i(23 downto 16));
                        l_inputs(7) := signed(img_data_i(31 downto 24));

                        for k in 0 to N_PAR-1 loop
                            pipe_prods(k) <= resize(l_inputs(k), 16) *
                                             resize(w1_dout(k),  16);
                        end loop;
                        state <= S_L1_ACCUM;

                    when S_L1_ACCUM =>
                        v_sum01 := pipe_prods(0) + pipe_prods(1);
                        v_sum23 := pipe_prods(2) + pipe_prods(3);
                        v_sum45 := pipe_prods(4) + pipe_prods(5);
                        v_sum67 := pipe_prods(6) + pipe_prods(7);
                        v_sum_total := (v_sum01 + v_sum23) + (v_sum45 + v_sum67);

                        acc_reg <= acc_reg + v_sum_total;

                        if l1_block_idx < L1_IN_BLOCKS-1 then
                            l1_block_idx <= l1_block_idx + 1;
                            w1_addr      <= weight_addr_l1(l1_neuron_idx, l1_block_idx + 1);
                            img_addr_o   <= to_unsigned(2*(l1_block_idx+1), 16);
                            state        <= S_L1_FETCH_A;
                        else
                            state <= S_L1_QUANT_CALC;
                        end if;

                    when S_L1_QUANT_CALC =>
                        mult_q_reg <= acc_reg * QPARAMS(1).M;
                        state      <= S_L1_QUANT_APPLY;

                    when S_L1_QUANT_APPLY =>
                        rounding := (others => '0');
                        if QPARAMS(1).shift > 0 then
                            rounding(QPARAMS(1).shift-1) := '1';
                        end if;
                        q64 := mult_q_reg + rounding;
                        q64 := shift_right(q64, QPARAMS(1).shift);
                        q64 := q64 + resize(QPARAMS(1).zp_out, 64);

                        act_l1(l1_neuron_idx) <= sat_int8(resize(q64, 32));

                        if l1_neuron_idx < L1_OUT_SIZE-1 then
                            l1_neuron_idx <= l1_neuron_idx + 1;
                            state         <= S_L1_SETUP;
                        else
                            l2_neuron_idx <= 0;
                            state         <= S_L2_SETUP;
                        end if;

                    -- LAYER 2
                    when S_L2_SETUP =>
                        b2_addr      <= to_unsigned(l2_neuron_idx, B2_ADDR_WIDTH);
                        w2_addr      <= weight_addr_l2(l2_neuron_idx, 0);
                        l2_block_idx <= 0;
                        state        <= S_L2_BIAS_WAIT;

                    when S_L2_BIAS_WAIT =>
                        bias_reg <= b2_dout;
                        acc_reg  <= b2_dout;
                        state    <= S_L2_FETCH;

                    when S_L2_FETCH =>
                        for k in 0 to N_PAR-1 loop
                            base_idx := l2_block_idx * N_PAR + k;
                            l_inputs(k) := act_l1(base_idx);
                            pipe_prods(k) <= resize(l_inputs(k), 16) *
                                             resize(w2_dout(k),  16);
                        end loop;
                        state <= S_L2_ACCUM;

                    when S_L2_ACCUM =>
                        v_sum01 := pipe_prods(0) + pipe_prods(1);
                        v_sum23 := pipe_prods(2) + pipe_prods(3);
                        v_sum45 := pipe_prods(4) + pipe_prods(5);
                        v_sum67 := pipe_prods(6) + pipe_prods(7);
                        v_sum_total := (v_sum01 + v_sum23) + (v_sum45 + v_sum67);

                        acc_reg <= acc_reg + v_sum_total;

                        if l2_block_idx < L2_IN_BLOCKS-1 then
                            l2_block_idx <= l2_block_idx + 1;
                            w2_addr      <= weight_addr_l2(l2_neuron_idx, l2_block_idx + 1);
                            state        <= S_L2_FETCH;
                        else
                            state <= S_L2_QUANT_CALC;
                        end if;

                    when S_L2_QUANT_CALC =>
                        mult_q_reg <= acc_reg * QPARAMS(2).M;
                        state      <= S_L2_QUANT_APPLY;

                    when S_L2_QUANT_APPLY =>
                        rounding := (others => '0');
                        if QPARAMS(2).shift > 0 then
                            rounding(QPARAMS(2).shift-1) := '1';
                        end if;
                        q64 := mult_q_reg + rounding;
                        q64 := shift_right(q64, QPARAMS(2).shift);
                        q64 := q64 + resize(QPARAMS(2).zp_out, 64);

                        act_l2(l2_neuron_idx) <= sat_int8(resize(q64, 32));

                        if l2_neuron_idx < L2_OUT_SIZE-1 then
                            l2_neuron_idx <= l2_neuron_idx + 1;
                            state         <= S_L2_SETUP;
                        else
                            l3_neuron_idx <= 0;
                            state         <= S_L3_SETUP;
                        end if;

                    -- LAYER 3
                    when S_L3_SETUP =>
                        b3_addr      <= to_unsigned(l3_neuron_idx, B3_ADDR_WIDTH);
                        w3_addr      <= weight_addr_l3(l3_neuron_idx, 0);
                        l3_block_idx <= 0;
                        state        <= S_L3_BIAS_WAIT;

                    when S_L3_BIAS_WAIT =>
                        bias_reg <= b3_dout;
                        acc_reg  <= b3_dout;
                        state    <= S_L3_FETCH;

                    when S_L3_FETCH =>
                        for k in 0 to N_PAR-1 loop
                            base_idx := l3_block_idx * N_PAR + k;
                            l_inputs(k) := act_l2(base_idx);
                            pipe_prods(k) <= resize(l_inputs(k), 16) *
                                             resize(w3_dout(k),  16);
                        end loop;
                        state <= S_L3_ACCUM;

                    when S_L3_ACCUM =>
                        v_sum01 := pipe_prods(0) + pipe_prods(1);
                        v_sum23 := pipe_prods(2) + pipe_prods(3);
                        v_sum45 := pipe_prods(4) + pipe_prods(5);
                        v_sum67 := pipe_prods(6) + pipe_prods(7);
                        v_sum_total := (v_sum01 + v_sum23) + (v_sum45 + v_sum67);

                        acc_reg <= acc_reg + v_sum_total;

                        if l3_block_idx < L3_IN_BLOCKS-1 then
                            l3_block_idx <= l3_block_idx + 1;
                            w3_addr      <= weight_addr_l3(l3_neuron_idx, l3_block_idx + 1);
                            state        <= S_L3_FETCH;
                        else
                            state <= S_L3_QUANT_CALC;
                        end if;

                    when S_L3_QUANT_CALC =>
                        mult_q_reg <= acc_reg * QPARAMS(3).M;
                        state      <= S_L3_QUANT_APPLY;

                    when S_L3_QUANT_APPLY =>
                        rounding := (others => '0');
                        if QPARAMS(3).shift > 0 then
                            rounding(QPARAMS(3).shift-1) := '1';
                        end if;
                        q64 := mult_q_reg + rounding;
                        q64 := shift_right(q64, QPARAMS(3).shift);
                        q64 := q64 + resize(QPARAMS(3).zp_out, 64);

                        act_l3(l3_neuron_idx) <= sat_int8(resize(q64, 32));

                        if l3_neuron_idx < L3_OUT_SIZE-1 then
                            l3_neuron_idx <= l3_neuron_idx + 1;
                            state         <= S_L3_SETUP;
                        else
                            l4_neuron_idx <= 0;
                            state         <= S_L4_SETUP;
                        end if;

                    -- LAYER 4
                    when S_L4_SETUP =>
                        b4_addr      <= to_unsigned(l4_neuron_idx, B4_ADDR_WIDTH);
                        w4_addr      <= weight_addr_l4(l4_neuron_idx, 0);
                        l4_block_idx <= 0;
                        state        <= S_L4_BIAS_WAIT;

                    when S_L4_BIAS_WAIT =>
                        bias_reg <= b4_dout;
                        acc_reg  <= b4_dout;
                        state    <= S_L4_FETCH;

                    when S_L4_FETCH =>
                        for k in 0 to N_PAR-1 loop
                            base_idx := l4_block_idx * N_PAR + k;
                            l_inputs(k) := act_l3(base_idx);
                            pipe_prods(k) <= resize(l_inputs(k), 16) *
                                             resize(w4_dout(k),  16);
                        end loop;
                        state <= S_L4_ACCUM;

                    when S_L4_ACCUM =>
                        v_sum01 := pipe_prods(0) + pipe_prods(1);
                        v_sum23 := pipe_prods(2) + pipe_prods(3);
                        v_sum45 := pipe_prods(4) + pipe_prods(5);
                        v_sum67 := pipe_prods(6) + pipe_prods(7);
                        v_sum_total := (v_sum01 + v_sum23) + (v_sum45 + v_sum67);

                        acc_reg <= acc_reg + v_sum_total;

                        if l4_block_idx < L4_IN_BLOCKS-1 then
                            l4_block_idx <= l4_block_idx + 1;
                            w4_addr      <= weight_addr_l4(l4_neuron_idx, l4_block_idx + 1);
                            state        <= S_L4_FETCH;
                        else
                            state <= S_L4_QUANT_CALC;
                        end if;

                    when S_L4_QUANT_CALC =>
                        mult_q_reg <= acc_reg * QPARAMS(4).M;
                        state      <= S_L4_QUANT_APPLY;

                    when S_L4_QUANT_APPLY =>
                        rounding := (others => '0');
                        if QPARAMS(4).shift > 0 then
                            rounding(QPARAMS(4).shift-1) := '1';
                        end if;
                        q64 := mult_q_reg + rounding;
                        q64 := shift_right(q64, QPARAMS(4).shift);
                        q64 := q64 + resize(QPARAMS(4).zp_out, 64);

                        act_l4(l4_neuron_idx) <= sat_int8(resize(q64, 32));

                        if l4_neuron_idx < L4_OUT_SIZE-1 then
                            l4_neuron_idx <= l4_neuron_idx + 1;
                            state         <= S_L4_SETUP;
                        else
                            state         <= S_DONE;
                        end if;

                    -- DONE / ERROR
                    when S_DONE =>
                        busy_reg <= '0';
                        done_reg <= '1';
                        state    <= S_IDLE;

                    when S_ERROR =>
                        error_reg <= '1';
                        busy_reg  <= '0';
                        state     <= S_IDLE;

                    when others =>
                        error_reg <= '1';
                        busy_reg  <= '0';
                        state     <= S_IDLE;
                end case;
            end if;
        end if;
    end process;

end architecture rtl;
