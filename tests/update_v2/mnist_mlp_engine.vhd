library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mnist_mlp_pkg.all;

entity mnist_mlp_engine is
    port (
        clk        : in  std_logic;
        rst_n      : in  std_logic;  -- active-low, synchronous
        start_i    : in  std_logic;
        busy_o     : out std_logic;
        done_o     : out std_logic;
        error_o    : out std_logic;

        -- Input image (flattened 28x28, 784 INT8)
        image_in   : in  int8_vector_t(0 to L0_SIZE-1);

        -- Output logits (10 INT8)
        logits_out : out int8_vector_t(0 to L4_OUT_SIZE-1);

        -- Weight ROMs
        w1_addr : out unsigned(W1_ADDR_WIDTH-1 downto 0);
        w1_dout : in  weight_word_t;
        w2_addr : out unsigned(W2_ADDR_WIDTH-1 downto 0);
        w2_dout : in  weight_word_t;
        w3_addr : out unsigned(W3_ADDR_WIDTH-1 downto 0);
        w3_dout : in  weight_word_t;
        w4_addr : out unsigned(W4_ADDR_WIDTH-1 downto 0);
        w4_dout : in  weight_word_t;

        -- Bias ROMs
        b1_addr : out unsigned(B1_ADDR_WIDTH-1 downto 0);
        b1_dout : in  int32_t;
        b2_addr : out unsigned(B2_ADDR_WIDTH-1 downto 0);
        b2_dout : in  int32_t;
        b3_addr : out unsigned(B3_ADDR_WIDTH-1 downto 0);
        b3_dout : in  int32_t;
        b4_addr : out unsigned(B4_ADDR_WIDTH-1 downto 0);
        b4_dout : in  int32_t
    );
end entity mnist_mlp_engine;


architecture rtl of mnist_mlp_engine is

    -- Activation buffers
    signal act_l1 : int8_vector_t(0 to L1_OUT_SIZE-1);
    signal act_l2 : int8_vector_t(0 to L2_OUT_SIZE-1);
    signal act_l3 : int8_vector_t(0 to L3_OUT_SIZE-1);
    signal act_l4 : int8_vector_t(0 to L4_OUT_SIZE-1);

    -- FSM states
    type state_t is (
        S_IDLE,

        -- Layer 1
        S_L1_SETUP_NEURON,
        S_L1_BIAS_WAIT,
        S_L1_MAC,
        S_L1_QUANT,

        -- Layer 2
        S_L2_SETUP_NEURON,
        S_L2_BIAS_WAIT,
        S_L2_MAC,
        S_L2_QUANT,

        -- Layer 3
        S_L3_SETUP_NEURON,
        S_L3_BIAS_WAIT,
        S_L3_MAC,
        S_L3_QUANT,

        -- Layer 4
        S_L4_SETUP_NEURON,
        S_L4_BIAS_WAIT,
        S_L4_MAC,
        S_L4_QUANT,

        S_DONE,
        S_ERROR
    );

    signal state : state_t := S_IDLE;

    -- Indices / counters
    signal l1_neuron_idx : integer range 0 to L1_OUT_SIZE-1 := 0;
    signal l1_block_idx  : integer range 0 to L1_IN_BLOCKS-1 := 0;

    signal l2_neuron_idx : integer range 0 to L2_OUT_SIZE-1 := 0;
    signal l2_block_idx  : integer range 0 to L2_IN_BLOCKS-1 := 0;

    signal l3_neuron_idx : integer range 0 to L3_OUT_SIZE-1 := 0;
    signal l3_block_idx  : integer range 0 to L3_IN_BLOCKS-1 := 0;

    signal l4_neuron_idx : integer range 0 to L4_OUT_SIZE-1 := 0;
    signal l4_block_idx  : integer range 0 to L4_IN_BLOCKS-1 := 0;

    -- Accumulator and bias
    signal acc_reg  : int32_t := (others => '0');
    signal bias_reg : int32_t := (others => '0');

    -- ROM address registers
    signal w1_addr_reg : unsigned(W1_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal w2_addr_reg : unsigned(W2_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal w3_addr_reg : unsigned(W3_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal w4_addr_reg : unsigned(W4_ADDR_WIDTH-1 downto 0) := (others => '0');

    signal b1_addr_reg : unsigned(B1_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal b2_addr_reg : unsigned(B2_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal b3_addr_reg : unsigned(B3_ADDR_WIDTH-1 downto 0) := (others => '0');
    signal b4_addr_reg : unsigned(B4_ADDR_WIDTH-1 downto 0) := (others => '0');

    -- Status
    signal busy_reg  : std_logic := '0';
    signal done_reg  : std_logic := '0';
    signal error_reg : std_logic := '0';

begin

    -- Port mapping to top-level
    w1_addr <= w1_addr_reg;
    w2_addr <= w2_addr_reg;
    w3_addr <= w3_addr_reg;
    w4_addr <= w4_addr_reg;

    b1_addr <= b1_addr_reg;
    b2_addr <= b2_addr_reg;
    b3_addr <= b3_addr_reg;
    b4_addr <= b4_addr_reg;

    busy_o  <= busy_reg;
    done_o  <= done_reg;
    error_o <= error_reg;

    logits_out <= act_l4;

    -- Main sequential process
    process (clk)
        -- Local variables
        variable k        : integer;
        variable base_idx : integer;

        variable mul       : int16_t;  -- 8x8 -> 16 bits
        variable sum_block : int32_t;

        variable q64   : signed(63 downto 0);
        variable tmp32 : int32_t;
    begin
        if rising_edge(clk) then
            if rst_n = '0' then
                state         <= S_IDLE;

                l1_neuron_idx <= 0;
                l1_block_idx  <= 0;
                l2_neuron_idx <= 0;
                l2_block_idx  <= 0;
                l3_neuron_idx <= 0;
                l3_block_idx  <= 0;
                l4_neuron_idx <= 0;
                l4_block_idx  <= 0;

                acc_reg       <= (others => '0');
                bias_reg      <= (others => '0');

                w1_addr_reg   <= (others => '0');
                w2_addr_reg   <= (others => '0');
                w3_addr_reg   <= (others => '0');
                w4_addr_reg   <= (others => '0');

                b1_addr_reg   <= (others => '0');
                b2_addr_reg   <= (others => '0');
                b3_addr_reg   <= (others => '0');
                b4_addr_reg   <= (others => '0');

                busy_reg      <= '0';
                done_reg      <= '0';
                error_reg     <= '0';

                act_l1        <= (others => (others => '0'));
                act_l2        <= (others => (others => '0'));
                act_l3        <= (others => (others => '0'));
                act_l4        <= (others => (others => '0'));

            else
                -- Default strobes
                done_reg <= '0';

                case state is

                    -- IDLE
                    when S_IDLE =>
                        busy_reg  <= '0';
                        error_reg <= '0';

                        if start_i = '1' then
                            busy_reg      <= '1';
                            l1_neuron_idx <= 0;
                            l1_block_idx  <= 0;
                            state         <= S_L1_SETUP_NEURON;
                        end if;

                    -- LAYER 1
                    when S_L1_SETUP_NEURON =>
                        b1_addr_reg   <= to_unsigned(l1_neuron_idx, B1_ADDR_WIDTH);
                        acc_reg       <= (others => '0');
                        l1_block_idx  <= 0;
                        w1_addr_reg   <= weight_addr_l1(l1_neuron_idx, 0);
                        state         <= S_L1_BIAS_WAIT;

                    when S_L1_BIAS_WAIT =>
                        bias_reg      <= b1_dout;
                        acc_reg       <= b1_dout;
                        l1_block_idx  <= 0;
                        state         <= S_L1_MAC;

                    when S_L1_MAC =>
                        sum_block := (others => '0');

                        for k in 0 to N_PAR-1 loop
                            base_idx := l1_block_idx * N_PAR + k;

                            -- 8x8 -> 16-bit product
                            mul := image_in(base_idx) * w1_dout(k);

                            -- accumulate into 32 bits
                            sum_block := sum_block + resize(mul, 32);
                        end loop;

                        acc_reg <= acc_reg + sum_block;

                        if l1_block_idx < L1_IN_BLOCKS-1 then
                            l1_block_idx <= l1_block_idx + 1;
                            w1_addr_reg  <= weight_addr_l1(
                                l1_neuron_idx,
                                l1_block_idx + 1
                            );
                            state        <= S_L1_MAC;
                        else
                            state <= S_L1_QUANT;
                        end if;

                    when S_L1_QUANT =>
                        -- (acc * M) >> shift + zp_out
                        q64   := shift_right(acc_reg * QPARAMS(1).M,
                                             QPARAMS(1).shift);
                        q64   := q64 + resize(QPARAMS(1).zp_out, 64);
                        tmp32 := resize(q64, 32);

                        act_l1(l1_neuron_idx) <= sat_int8(tmp32);

                        if l1_neuron_idx < L1_OUT_SIZE-1 then
                            l1_neuron_idx <= l1_neuron_idx + 1;
                            state         <= S_L1_SETUP_NEURON;
                        else
                            l2_neuron_idx <= 0;
                            l2_block_idx  <= 0;
                            state         <= S_L2_SETUP_NEURON;
                        end if;

                    -- LAYER 2
                    when S_L2_SETUP_NEURON =>
                        b2_addr_reg   <= to_unsigned(l2_neuron_idx, B2_ADDR_WIDTH);
                        acc_reg       <= (others => '0');
                        l2_block_idx  <= 0;
                        w2_addr_reg   <= weight_addr_l2(l2_neuron_idx, 0);
                        state         <= S_L2_BIAS_WAIT;

                    when S_L2_BIAS_WAIT =>
                        bias_reg      <= b2_dout;
                        acc_reg       <= b2_dout;
                        l2_block_idx  <= 0;
                        state         <= S_L2_MAC;

                    when S_L2_MAC =>
                        sum_block := (others => '0');

                        for k in 0 to N_PAR-1 loop
                            base_idx := l2_block_idx * N_PAR + k;

                            mul := act_l1(base_idx) * w2_dout(k);

                            sum_block := sum_block + resize(mul, 32);
                        end loop;

                        acc_reg <= acc_reg + sum_block;

                        if l2_block_idx < L2_IN_BLOCKS-1 then
                            l2_block_idx <= l2_block_idx + 1;
                            w2_addr_reg  <= weight_addr_l2(
                                l2_neuron_idx,
                                l2_block_idx + 1
                            );
                            state        <= S_L2_MAC;
                        else
                            state <= S_L2_QUANT;
                        end if;

                    when S_L2_QUANT =>
                        q64   := shift_right(acc_reg * QPARAMS(2).M,
                                             QPARAMS(2).shift);
                        q64   := q64 + resize(QPARAMS(2).zp_out, 64);
                        tmp32 := resize(q64, 32);

                        act_l2(l2_neuron_idx) <= sat_int8(tmp32);

                        if l2_neuron_idx < L2_OUT_SIZE-1 then
                            l2_neuron_idx <= l2_neuron_idx + 1;
                            state         <= S_L2_SETUP_NEURON;
                        else
                            l3_neuron_idx <= 0;
                            l3_block_idx  <= 0;
                            state         <= S_L3_SETUP_NEURON;
                        end if;

                    -- LAYER 3
                    when S_L3_SETUP_NEURON =>
                        b3_addr_reg   <= to_unsigned(l3_neuron_idx, B3_ADDR_WIDTH);
                        acc_reg       <= (others => '0');
                        l3_block_idx  <= 0;
                        w3_addr_reg   <= weight_addr_l3(l3_neuron_idx, 0);
                        state         <= S_L3_BIAS_WAIT;

                    when S_L3_BIAS_WAIT =>
                        bias_reg      <= b3_dout;
                        acc_reg       <= b3_dout;
                        l3_block_idx  <= 0;
                        state         <= S_L3_MAC;

                    when S_L3_MAC =>
                        sum_block := (others => '0');

                        for k in 0 to N_PAR-1 loop
                            base_idx := l3_block_idx * N_PAR + k;

                            mul := act_l2(base_idx) * w3_dout(k);

                            sum_block := sum_block + resize(mul, 32);
                        end loop;

                        acc_reg <= acc_reg + sum_block;

                        if l3_block_idx < L3_IN_BLOCKS-1 then
                            l3_block_idx <= l3_block_idx + 1;
                            w3_addr_reg  <= weight_addr_l3(
                                l3_neuron_idx,
                                l3_block_idx + 1
                            );
                            state        <= S_L3_MAC;
                        else
                            state <= S_L3_QUANT;
                        end if;

                    when S_L3_QUANT =>
                        q64   := shift_right(acc_reg * QPARAMS(3).M,
                                             QPARAMS(3).shift);
                        q64   := q64 + resize(QPARAMS(3).zp_out, 64);
                        tmp32 := resize(q64, 32);

                        act_l3(l3_neuron_idx) <= sat_int8(tmp32);

                        if l3_neuron_idx < L3_OUT_SIZE-1 then
                            l3_neuron_idx <= l3_neuron_idx + 1;
                            state         <= S_L3_SETUP_NEURON;
                        else
                            l4_neuron_idx <= 0;
                            l4_block_idx  <= 0;
                            state         <= S_L4_SETUP_NEURON;
                        end if;

                    -- LAYER 4
                    when S_L4_SETUP_NEURON =>
                        b4_addr_reg   <= to_unsigned(l4_neuron_idx, B4_ADDR_WIDTH);
                        acc_reg       <= (others => '0');
                        l4_block_idx  <= 0;
                        w4_addr_reg   <= weight_addr_l4(l4_neuron_idx, 0);
                        state         <= S_L4_BIAS_WAIT;

                    when S_L4_BIAS_WAIT =>
                        bias_reg      <= b4_dout;
                        acc_reg       <= b4_dout;
                        l4_block_idx  <= 0;
                        state         <= S_L4_MAC;

                    when S_L4_MAC =>
                        sum_block := (others => '0');

                        for k in 0 to N_PAR-1 loop
                            base_idx := l4_block_idx * N_PAR + k;

                            mul := act_l3(base_idx) * w4_dout(k);

                            sum_block := sum_block + resize(mul, 32);
                        end loop;

                        acc_reg <= acc_reg + sum_block;

                        if l4_block_idx < L4_IN_BLOCKS-1 then
                            l4_block_idx <= l4_block_idx + 1;
                            w4_addr_reg  <= weight_addr_l4(
                                l4_neuron_idx,
                                l4_block_idx + 1
                            );
                            state        <= S_L4_MAC;
                        else
                            state <= S_L4_QUANT;
                        end if;

                    when S_L4_QUANT =>
                        q64   := shift_right(acc_reg * QPARAMS(4).M,
                                             QPARAMS(4).shift);
                        q64   := q64 + resize(QPARAMS(4).zp_out, 64);
                        tmp32 := resize(q64, 32);

                        act_l4(l4_neuron_idx) <= sat_int8(tmp32);

                        if l4_neuron_idx < L4_OUT_SIZE-1 then
                            l4_neuron_idx <= l4_neuron_idx + 1;
                            state         <= S_L4_SETUP_NEURON;
                        else
                            state <= S_DONE;
                        end if;

                    when S_DONE =>
                        busy_reg <= '0';
                        done_reg <= '1';
                        state    <= S_IDLE;

                    when S_ERROR =>
                        busy_reg  <= '0';
                        error_reg <= '1';
                        state     <= S_IDLE;

                end case;
            end if;
        end if;
    end process;

end architecture rtl;
