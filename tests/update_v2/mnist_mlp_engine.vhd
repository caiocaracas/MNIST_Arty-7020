library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mnist_mlp_pkg.all;

entity mnist_mlp_engine is
    port (
        clk   : in  std_logic;
        rst_n : in  std_logic;

        start_i : in  std_logic;
        busy_o  : out std_logic;
        done_o  : out std_logic;
        error_o : out std_logic;

        -- Input image
        image_in   : in  int8_vector_t(0 to L0_SIZE-1);

        -- Output logits
        logits_out : out int8_vector_t(0 to L4_SIZE-1);

        -- Weight ROM interfaces (each word has N_PAR int8 weights)
        -- Layer 1
        w1_addr : out unsigned(W1_ADDR_WIDTH-1 downto 0);
        w1_dout : in  weight_word_t;

        -- Layer 2
        w2_addr : out unsigned(W2_ADDR_WIDTH-1 downto 0);
        w2_dout : in  weight_word_t;

        -- Layer 3
        w3_addr : out unsigned(W3_ADDR_WIDTH-1 downto 0);
        w3_dout : in  weight_word_t;

        -- Layer 4
        w4_addr : out unsigned(W4_ADDR_WIDTH-1 downto 0);
        w4_dout : in  weight_word_t;

        -- Bias ROM interfaces (one bias per output neuron)
        -- Layer 1
        b1_addr : out unsigned(B1_ADDR_WIDTH-1 downto 0);
        b1_dout : in  int32_t;

        -- Layer 2
        b2_addr : out unsigned(B2_ADDR_WIDTH-1 downto 0);
        b2_dout : in  int32_t;

        -- Layer 3
        b3_addr : out unsigned(B3_ADDR_WIDTH-1 downto 0);
        b3_dout : in  int32_t;

        -- Layer 4
        b4_addr : out unsigned(B4_ADDR_WIDTH-1 downto 0);
        b4_dout : in  int32_t
    );
end entity mnist_mlp_engine;

architecture rtl of mnist_mlp_engine is

    type state_t is (
        S_IDLE,
        S_L1_INIT,  S_L1_ACCUM,  S_L1_QUANT,
        S_L2_INIT,  S_L2_ACCUM,  S_L2_QUANT,
        S_L3_INIT,  S_L3_ACCUM,  S_L3_QUANT,
        S_L4_INIT,  S_L4_ACCUM,  S_L4_QUANT,
        S_DONE,
        S_ERROR
    );

    signal state_reg, state_next : state_t;

    -- Indices (0 .. OUT_SIZE-1 / 0 .. IN_BLOCKS-1)
    signal l1_neuron_idx : integer range 0 to L1_OUT_SIZE-1 := 0;
    signal l1_block_idx  : integer range 0 to L1_IN_BLOCKS-1 := 0;

    signal l2_neuron_idx : integer range 0 to L2_OUT_SIZE-1 := 0;
    signal l2_block_idx  : integer range 0 to L2_IN_BLOCKS-1 := 0;

    signal l3_neuron_idx : integer range 0 to L3_OUT_SIZE-1 := 0;
    signal l3_block_idx  : integer range 0 to L3_IN_BLOCKS-1 := 0;

    signal l4_neuron_idx : integer range 0 to L4_OUT_SIZE-1 := 0;
    signal l4_block_idx  : integer range 0 to L4_IN_BLOCKS-1 := 0;

    -- Accumulator
    signal acc_reg : int32_t := (others => '0');

    -- Activation buffers for layers 1..4
    signal act_l1 : int8_vector_t(0 to L1_OUT_SIZE-1);
    signal act_l2 : int8_vector_t(0 to L2_OUT_SIZE-1);
    signal act_l3 : int8_vector_t(0 to L3_OUT_SIZE-1);
    signal act_l4 : int8_vector_t(0 to L4_OUT_SIZE-1);

    -- Status registers
    signal busy_reg  : std_logic := '0';
    signal done_reg  : std_logic := '0';
    signal error_reg : std_logic := '0';

begin

    busy_o    <= busy_reg;
    done_o    <= done_reg;
    error_o   <= error_reg;
    logits_out <= act_l4;

    process (clk, rst_n)
    begin
        if rst_n = '0' then
            state_reg <= S_IDLE;
        elsif rising_edge(clk) then
            state_reg <= state_next;
        end if;
    end process;

    process (state_reg,
             start_i,
             l1_neuron_idx, l1_block_idx,
             l2_neuron_idx, l2_block_idx,
             l3_neuron_idx, l3_block_idx,
             l4_neuron_idx, l4_block_idx)
    begin
        state_next <= state_reg;

        case state_reg is

            when S_IDLE =>
                if start_i = '1' then
                    state_next <= S_L1_INIT;
                end if;

            -- Layer 1
            when S_L1_INIT =>
                state_next <= S_L1_ACCUM;

            when S_L1_ACCUM =>
                if l1_block_idx = L1_IN_BLOCKS-1 then
                    state_next <= S_L1_QUANT;
                end if;

            when S_L1_QUANT =>
                if l1_neuron_idx = L1_OUT_SIZE-1 then
                    state_next <= S_L2_INIT;
                else
                    state_next <= S_L1_INIT;
                end if;

            -- Layer 2
            when S_L2_INIT =>
                state_next <= S_L2_ACCUM;

            when S_L2_ACCUM =>
                if l2_block_idx = L2_IN_BLOCKS-1 then
                    state_next <= S_L2_QUANT;
                end if;

            when S_L2_QUANT =>
                if l2_neuron_idx = L2_OUT_SIZE-1 then
                    state_next <= S_L3_INIT;
                else
                    state_next <= S_L2_INIT;
                end if;

            -- Layer 3
            when S_L3_INIT =>
                state_next <= S_L3_ACCUM;

            when S_L3_ACCUM =>
                if l3_block_idx = L3_IN_BLOCKS-1 then
                    state_next <= S_L3_QUANT;
                end if;

            when S_L3_QUANT =>
                if l3_neuron_idx = L3_OUT_SIZE-1 then
                    state_next <= S_L4_INIT;
                else
                    state_next <= S_L3_INIT;
                end if;

            -- Layer 4
            when S_L4_INIT =>
                state_next <= S_L4_ACCUM;

            when S_L4_ACCUM =>
                if l4_block_idx = L4_IN_BLOCKS-1 then
                    state_next <= S_L4_QUANT;
                end if;

            when S_L4_QUANT =>
                if l4_neuron_idx = L4_OUT_SIZE-1 then
                    state_next <= S_DONE;
                else
                    state_next <= S_L4_INIT;
                end if;

            when S_DONE =>
                state_next <= S_IDLE;

            when S_ERROR =>
                state_next <= S_IDLE;

            when others =>
                state_next <= S_ERROR;

        end case;
    end process;

    process (clk, rst_n)
        variable mul      : int16_t;
        variable sum_block: int32_t;
        variable q_acc    : int32_t;
        variable act_val  : int8_t;
        variable k        : integer;
    begin
        if rst_n = '0' then
            busy_reg   <= '0';
            done_reg   <= '0';
            error_reg  <= '0';
            acc_reg    <= (others => '0');

            l1_neuron_idx <= 0;
            l1_block_idx  <= 0;
            l2_neuron_idx <= 0;
            l2_block_idx  <= 0;
            l3_neuron_idx <= 0;
            l3_block_idx  <= 0;
            l4_neuron_idx <= 0;
            l4_block_idx  <= 0;

            act_l1 <= (others => to_signed(0,8));
            act_l2 <= (others => to_signed(0,8));
            act_l3 <= (others => to_signed(0,8));
            act_l4 <= (others => to_signed(0,8));

            w1_addr <= (others => '0');
            w2_addr <= (others => '0');
            w3_addr <= (others => '0');
            w4_addr <= (others => '0');
            b1_addr <= (others => '0');
            b2_addr <= (others => '0');
            b3_addr <= (others => '0');
            b4_addr <= (others => '0');

        elsif rising_edge(clk) then

            done_reg <= '0';

            case state_reg is

                -- IDLE
                when S_IDLE =>
                    busy_reg <= '0';
                    acc_reg  <= (others => '0');
                    if start_i = '1' then
                        busy_reg      <= '1';
                        l1_neuron_idx <= 0;
                        l1_block_idx  <= 0;
                        b1_addr <= to_unsigned(0, B1_ADDR_WIDTH);
                        w1_addr <= weight_addr_l1(0, 0);
                    end if;

                -- Layer 1
                when S_L1_INIT =>
                    b1_addr <= to_unsigned(l1_neuron_idx, B1_ADDR_WIDTH);
                    acc_reg <= b1_dout;
                    l1_block_idx <= 0;
                    w1_addr <= weight_addr_l1(l1_neuron_idx, 0);

                when S_L1_ACCUM =>
                    sum_block := (others => '0');

                    for i in 0 to N_PAR-1 loop
                        k := l1_block_idx * N_PAR + i;
                        if k < L1_IN_SIZE then
                            act_val := image_in(k);
                            -- 8x8 signed multiply -> 16 bits, DSP-friendly
                            mul := image_in(k) * w1_dout(i);
                            sum_block := sum_block + resize(mul, sum_block'length);
                        end if;
                    end loop;

                    acc_reg <= acc_reg + sum_block;

                    if l1_block_idx < L1_IN_BLOCKS-1 then
                        l1_block_idx <= l1_block_idx + 1;
                        w1_addr <= weight_addr_l1(l1_neuron_idx, l1_block_idx + 1);
                    end if;

                when S_L1_QUANT =>
                    q_acc := acc_reg * QPARAMS(1).M;
                    if QPARAMS(1).shift > 0 then
                        q_acc := shift_right(q_acc, QPARAMS(1).shift);
                    end if;
                    q_acc := q_acc + resize(QPARAMS(1).zp_out, 32);

                    act_l1(l1_neuron_idx) <= sat_int8(q_acc);

                    if l1_neuron_idx < L1_OUT_SIZE-1 then
                        l1_neuron_idx <= l1_neuron_idx + 1;
                        l1_block_idx  <= 0;
                        acc_reg       <= (others => '0');
                        b1_addr       <= to_unsigned(l1_neuron_idx + 1, B1_ADDR_WIDTH);
                        w1_addr       <= weight_addr_l1(l1_neuron_idx + 1, 0);
                    else
                        l2_neuron_idx <= 0;
                        l2_block_idx  <= 0;
                    end if;

                -- Layer 2
                when S_L2_INIT =>
                    b2_addr <= to_unsigned(l2_neuron_idx, B2_ADDR_WIDTH);
                    acc_reg <= b2_dout;
                    l2_block_idx <= 0;
                    w2_addr <= weight_addr_l2(l2_neuron_idx, 0);

                when S_L2_ACCUM =>
                    sum_block := (others => '0');

                    for i in 0 to N_PAR-1 loop
                        k := l2_block_idx * N_PAR + i;
                        if k < L2_IN_SIZE then
                            act_val := act_l1(k);
                            mul := act_l1(k) * w2_dout(i);
                            sum_block := sum_block + resize(mul, sum_block'length);
                        end if;
                    end loop;

                    acc_reg <= acc_reg + sum_block;

                    if l2_block_idx < L2_IN_BLOCKS-1 then
                        l2_block_idx <= l2_block_idx + 1;
                        w2_addr <= weight_addr_l2(l2_neuron_idx, l2_block_idx + 1);
                    end if;

                when S_L2_QUANT =>
                    q_acc := acc_reg * QPARAMS(2).M;
                    if QPARAMS(2).shift > 0 then
                        q_acc := shift_right(q_acc, QPARAMS(2).shift);
                    end if;
                    q_acc := q_acc + resize(QPARAMS(2).zp_out, 32);

                    act_l2(l2_neuron_idx) <= sat_int8(q_acc);

                    if l2_neuron_idx < L2_OUT_SIZE-1 then
                        l2_neuron_idx <= l2_neuron_idx + 1;
                        l2_block_idx  <= 0;
                        acc_reg       <= (others => '0');
                        b2_addr       <= to_unsigned(l2_neuron_idx + 1, B2_ADDR_WIDTH);
                        w2_addr       <= weight_addr_l2(l2_neuron_idx + 1, 0);
                    else
                        l3_neuron_idx <= 0;
                        l3_block_idx  <= 0;
                    end if;

                -- Layer 3
                when S_L3_INIT =>
                    b3_addr <= to_unsigned(l3_neuron_idx, B3_ADDR_WIDTH);
                    acc_reg <= b3_dout;
                    l3_block_idx <= 0;
                    w3_addr <= weight_addr_l3(l3_neuron_idx, 0);

                when S_L3_ACCUM =>
                    sum_block := (others => '0');

                    for i in 0 to N_PAR-1 loop
                        k := l3_block_idx * N_PAR + i;
                        if k < L3_IN_SIZE then
                            act_val := act_l2(k);
                            mul := act_l2(k) * w3_dout(i);
                            sum_block := sum_block + resize(mul, sum_block'length);
                        end if;
                    end loop;

                    acc_reg <= acc_reg + sum_block;

                    if l3_block_idx < L3_IN_BLOCKS-1 then
                        l3_block_idx <= l3_block_idx + 1;
                        w3_addr <= weight_addr_l3(l3_neuron_idx, l3_block_idx + 1);
                    end if;

                when S_L3_QUANT =>
                    q_acc := acc_reg * QPARAMS(3).M;
                    if QPARAMS(3).shift > 0 then
                        q_acc := shift_right(q_acc, QPARAMS(3).shift);
                    end if;
                    q_acc := q_acc + resize(QPARAMS(3).zp_out, 32);

                    act_l3(l3_neuron_idx) <= sat_int8(q_acc);

                    if l3_neuron_idx < L3_OUT_SIZE-1 then
                        l3_neuron_idx <= l3_neuron_idx + 1;
                        l3_block_idx  <= 0;
                        acc_reg       <= (others => '0');
                        b3_addr       <= to_unsigned(l3_neuron_idx + 1, B3_ADDR_WIDTH);
                        w3_addr       <= weight_addr_l3(l3_neuron_idx + 1, 0);
                    else
                        l4_neuron_idx <= 0;
                        l4_block_idx  <= 0;
                    end if;

                -- Layer 4
                when S_L4_INIT =>
                    b4_addr <= to_unsigned(l4_neuron_idx, B4_ADDR_WIDTH);
                    acc_reg <= b4_dout;
                    l4_block_idx <= 0;
                    w4_addr <= weight_addr_l4(l4_neuron_idx, 0);

                when S_L4_ACCUM =>
                    sum_block := (others => '0');

                    for i in 0 to N_PAR-1 loop
                        k := l4_block_idx * N_PAR + i;
                        if k < L4_IN_SIZE then
                            act_val := act_l3(k);
                            mul := act_l3(k) * w4_dout(i);
                            sum_block := sum_block + resize(mul, sum_block'length);
                        end if;
                    end loop;

                    acc_reg <= acc_reg + sum_block;

                    if l4_block_idx < L4_IN_BLOCKS-1 then
                        l4_block_idx <= l4_block_idx + 1;
                        w4_addr <= weight_addr_l4(l4_neuron_idx, l4_block_idx + 1);
                    end if;

                when S_L4_QUANT =>
                    q_acc := acc_reg * QPARAMS(4).M;
                    if QPARAMS(4).shift > 0 then
                        q_acc := shift_right(q_acc, QPARAMS(4).shift);
                    end if;
                    q_acc := q_acc + resize(QPARAMS(4).zp_out, 32);

                    act_l4(l4_neuron_idx) <= sat_int8(q_acc);

                    if l4_neuron_idx < L4_OUT_SIZE-1 then
                        l4_neuron_idx <= l4_neuron_idx + 1;
                        l4_block_idx  <= 0;
                        acc_reg       <= (others => '0');
                        b4_addr       <= to_unsigned(l4_neuron_idx + 1, B4_ADDR_WIDTH);
                        w4_addr       <= weight_addr_l4(l4_neuron_idx + 1, 0);
                    end if;

                -- DONE / ERROR
                when S_DONE =>
                    busy_reg <= '0';
                    done_reg <= '1';

                when S_ERROR =>
                    busy_reg <= '0';
                    error_reg <= '1';

                when others =>
                    busy_reg <= '0';
                    error_reg <= '1';
            end case;
        end if;
    end process;
end architecture rtl;
