-- File: mnist_mlp_pkg.vhd
-- Global types and parameters for MNIST MLP INT8 accelerator
-- with input parallelism (N_PAR MACs per cycle).

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package mnist_mlp_pkg is

    subtype int8_t  is signed(7 downto 0);    -- 8-bit signed
    subtype int32_t is signed(31 downto 0);   -- 32-bit signed

    type int8_vector_t  is array (natural range <>) of int8_t;
    type int32_vector_t is array (natural range <>) of int32_t;

    -- Parallelism factor (number of MACs per cycle)
    -- N_PAR must divide all layer input sizes:
    --  - L0_SIZE (784)
    --  - L1_SIZE (256)
    --  - L2_SIZE (128)
    --  - L3_SIZE (64)
    constant N_PAR : integer := 8;

    constant L0_SIZE : integer := 784;  -- input layer 
    constant L1_SIZE : integer := 256;  -- hidden layer 1
    constant L2_SIZE : integer := 128;  -- hidden layer 2
    constant L3_SIZE : integer := 64;   -- hidden layer 3
    constant L4_SIZE : integer := 10;   -- output layer 

    -- Convenience aliases for layer input/output sizes
    constant L1_IN_SIZE  : integer := L0_SIZE;
    constant L1_OUT_SIZE : integer := L1_SIZE;

    constant L2_IN_SIZE  : integer := L1_SIZE;
    constant L2_OUT_SIZE : integer := L2_SIZE;

    constant L3_IN_SIZE  : integer := L2_SIZE;
    constant L3_OUT_SIZE : integer := L3_SIZE;

    constant L4_IN_SIZE  : integer := L3_SIZE;
    constant L4_OUT_SIZE : integer := L4_SIZE;

    -- Input blocks per layer (each block has N_PAR inputs)
    constant L1_IN_BLOCKS : integer := L1_IN_SIZE / N_PAR;
    constant L2_IN_BLOCKS : integer := L2_IN_SIZE / N_PAR;
    constant L3_IN_BLOCKS : integer := L3_IN_SIZE / N_PAR;
    constant L4_IN_BLOCKS : integer := L4_IN_SIZE / N_PAR;

    -- Weight word counts per layer
    -- Each ROM word contains N_PAR weights (int8_t).
    -- Total words = OUT_SIZE * IN_BLOCKS.
    constant W1_WORD_COUNT : integer := L1_OUT_SIZE * L1_IN_BLOCKS;
    constant W2_WORD_COUNT : integer := L2_OUT_SIZE * L2_IN_BLOCKS;
    constant W3_WORD_COUNT : integer := L3_OUT_SIZE * L3_IN_BLOCKS;
    constant W4_WORD_COUNT : integer := L4_OUT_SIZE * L4_IN_BLOCKS;

    -- Bias counts (1 bias per output neuron)
    constant B1_COUNT : integer := L1_OUT_SIZE;
    constant B2_COUNT : integer := L2_OUT_SIZE;
    constant B3_COUNT : integer := L3_OUT_SIZE;
    constant B4_COUNT : integer := L4_OUT_SIZE;

    -- Helper: ceiling log2 (for address width)
    function clog2 (
        n : integer
    ) return integer;

    -- Address widths for weight ROMs (word-based, not element-based)
    constant W1_ADDR_WIDTH : integer := clog2(W1_WORD_COUNT);
    constant W2_ADDR_WIDTH : integer := clog2(W2_WORD_COUNT);
    constant W3_ADDR_WIDTH : integer := clog2(W3_WORD_COUNT);
    constant W4_ADDR_WIDTH : integer := clog2(W4_WORD_COUNT);

    -- Address widths for bias ROMs
    constant B1_ADDR_WIDTH : integer := clog2(B1_COUNT);
    constant B2_ADDR_WIDTH : integer := clog2(B2_COUNT);
    constant B3_ADDR_WIDTH : integer := clog2(B3_COUNT);
    constant B4_ADDR_WIDTH : integer := clog2(B4_COUNT);

    -- Types for weight ROM word (N_PAR int8_t values)
    type weight_word_t is array (0 to N_PAR-1) of int8_t;

    -- Quantization parameters structure
    -- y_int8 = clip_int8( ((acc * M) >> SHIFT) + ZP_OUT )
    type layer_qparams_t is record
        M      : int32_t;  -- multiplier applied to 32-bit accumulator
        shift  : integer;  -- right shift amount
        zp_out : int8_t;   -- output zero-point for this layer's activations
    end record;

    type qparams_array_t is array (1 to 4) of layer_qparams_t;

    -- Default stub values. This constant is expected to be auto-generated from the Python quantization script
    constant QPARAMS : qparams_array_t := (
        1 => (M => to_signed(1,32), shift => 0, zp_out => to_signed(0,8)), -- L1
        2 => (M => to_signed(1,32), shift => 0, zp_out => to_signed(0,8)), -- L2
        3 => (M => to_signed(1,32), shift => 0, zp_out => to_signed(0,8)), -- L3
        4 => (M => to_signed(1,32), shift => 0, zp_out => to_signed(0,8))  -- L4
    );

    -- Helper functions: weight address mapping (block-based)
    -- For each layer: (neuron_idx, block_idx) -> ROM word address.
    -- Each block corresponds to N_PAR consecutive input indices.
    -- addr = neuron_idx * IN_BLOCKS + block_idx
    function weight_addr_l1 (
        neuron_idx : integer;
        block_idx  : integer
    ) return unsigned;

    function weight_addr_l2 (
        neuron_idx : integer;
        block_idx  : integer
    ) return unsigned;

    function weight_addr_l3 (
        neuron_idx : integer;
        block_idx  : integer
    ) return unsigned;

    function weight_addr_l4 (
        neuron_idx : integer;
        block_idx  : integer
    ) return unsigned;

    -- Helper function: saturate int32 to int8
    function sat_int8 (
        x : int32_t
    ) return int8_t;

end package mnist_mlp_pkg;

package body mnist_mlp_pkg is

    -- Ceiling log2
    function clog2 (
        n : integer
    ) return integer is
        variable r : integer := 0;
        variable v : integer := n - 1;
    begin
        while v > 0 loop
            v := v / 2;
            r := r + 1;
        end loop;
        return r;
    end function;

    -- Weight address mapping for each layer (block-based)
    function weight_addr_l1 (
        neuron_idx : integer;
        block_idx  : integer
    ) return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L1_IN_BLOCKS + block_idx;
        return resize(to_unsigned(lin, W1_ADDR_WIDTH), W1_ADDR_WIDTH);
    end function;

    function weight_addr_l2 (
        neuron_idx : integer;
        block_idx  : integer
    ) return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L2_IN_BLOCKS + block_idx;
        return resize(to_unsigned(lin, W2_ADDR_WIDTH), W2_ADDR_WIDTH);
    end function;

    function weight_addr_l3 (
        neuron_idx : integer;
        block_idx  : integer
    ) return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L3_IN_BLOCKS + block_idx;
        return resize(to_unsigned(lin, W3_ADDR_WIDTH), W3_ADDR_WIDTH);
    end function;

    function weight_addr_l4 (
        neuron_idx : integer;
        block_idx  : integer
    ) return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L4_IN_BLOCKS + block_idx;
        return resize(to_unsigned(lin, W4_ADDR_WIDTH), W4_ADDR_WIDTH);
    end function;

    -- Saturate int32_t to int8_t: clip to [-128, 127]
    function sat_int8 (
        x : int32_t
    ) return int8_t is
        variable y : int8_t;
    begin
        if x > to_signed(127, x'length) then
            y := to_signed(127, 8);
        elsif x < to_signed(-128, x'length) then
            y := to_signed(-128, 8);
        else
            y := resize(x, 8);
        end if;
        return y;
    end function;

end package body mnist_mlp_pkg;

