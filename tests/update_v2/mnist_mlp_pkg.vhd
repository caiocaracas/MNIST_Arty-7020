-- Global types and parameters for MNIST MLP INT8 accelerator

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package mnist_mlp_pkg is

    subtype int8_t  is signed(7 downto 0);    -- 8-bit signed for activations/weights
    subtype int32_t is signed(31 downto 0);   -- 32-bit signed for accumulators/bias

    type int8_vector_t  is array (natural range <>) of int8_t;
    type int32_vector_t is array (natural range <>) of int32_t;

    constant L0_SIZE : integer := 784;  -- input layer (pixels)
    constant L1_SIZE : integer := 256;  -- hidden layer 1
    constant L2_SIZE : integer := 128;  -- hidden layer 2
    constant L3_SIZE : integer := 64;   -- hidden layer 3
    constant L4_SIZE : integer := 10;   -- output layer

    constant L1_IN_SIZE  : integer := L0_SIZE;
    constant L1_OUT_SIZE : integer := L1_SIZE;
    constant L2_IN_SIZE  : integer := L1_SIZE;
    constant L2_OUT_SIZE : integer := L2_SIZE;
    constant L3_IN_SIZE  : integer := L2_SIZE;
    constant L3_OUT_SIZE : integer := L3_SIZE;
    constant L4_IN_SIZE  : integer := L3_SIZE;
    constant L4_OUT_SIZE : integer := L4_SIZE;

    -- Weight and bias counts per layer
    -- Total number of elements in each weight/bias tensor.
    constant W1_COUNT : integer := L1_OUT_SIZE * L1_IN_SIZE; -- 256 * 784
    constant W2_COUNT : integer := L2_OUT_SIZE * L2_IN_SIZE; -- 128 * 256
    constant W3_COUNT : integer := L3_OUT_SIZE * L3_IN_SIZE; -- 64  * 128
    constant W4_COUNT : integer := L4_OUT_SIZE * L4_IN_SIZE; -- 10  * 64

    constant B1_COUNT : integer := L1_OUT_SIZE; -- 256
    constant B2_COUNT : integer := L2_OUT_SIZE; -- 128
    constant B3_COUNT : integer := L3_OUT_SIZE; -- 64
    constant B4_COUNT : integer := L4_OUT_SIZE; -- 10

    -- W1_COUNT = 256 * 784 = 200704  -> 18 bits
    -- W2_COUNT = 128 * 256 = 32768   -> 15 bits
    -- W3_COUNT = 64  * 128 = 8192    -> 13 bits
    -- W4_COUNT = 10  * 64  = 640     -> 10 bits
    constant W1_ADDR_WIDTH : integer := 18;
    constant W2_ADDR_WIDTH : integer := 15;
    constant W3_ADDR_WIDTH : integer := 13;
    constant W4_ADDR_WIDTH : integer := 10;

    -- B1 = 256 -> 8 bits; B2 = 128 -> 7 bits; B3 = 64 -> 6 bits; B4 = 10 -> 4 bits.
    constant B1_ADDR_WIDTH : integer := 8;
    constant B2_ADDR_WIDTH : integer := 7;
    constant B3_ADDR_WIDTH : integer := 6;
    constant B4_ADDR_WIDTH : integer := 4;

    -- Quantization parameters structure
    -- y_int8 = clip_int8( ((acc * M) >> SHIFT) + ZP_OUT )
    type layer_qparams_t is record
        M      : int32_t;  -- multiplier applied to 32-bit accumulator
        shift  : integer;  -- right shift amount
        zp_out : int8_t;   -- output zero-point for this layer's activations
    end record;

    -- Array of quantization params per layer (1..4)
    type qparams_array_t is array (1 to 4) of layer_qparams_t;

    -- Default stub values. This constant is expected to be auto-generated from the python quantization script
    constant QPARAMS : qparams_array_t := (
        1 => (M => to_signed(1,32), shift => 0, zp_out => to_signed(0,8)), -- L1
        2 => (M => to_signed(1,32), shift => 0, zp_out => to_signed(0,8)), -- L2
        3 => (M => to_signed(1,32), shift => 0, zp_out => to_signed(0,8)), -- L3
        4 => (M => to_signed(1,32), shift => 0, zp_out => to_signed(0,8))  -- L4
    );

    -- Helper functions: weight address mapping
    -- Linearize (neuron_idx, input_idx) -> linear address for ROM
    -- addr = neuron_idx * input_size + input_idx
    function weight_addr_l1 (
        neuron_idx : integer;
        input_idx  : integer
    ) return unsigned;

    function weight_addr_l2 (
        neuron_idx : integer;
        input_idx  : integer
    ) return unsigned;

    function weight_addr_l3 (
        neuron_idx : integer;
        input_idx  : integer
    ) return unsigned;

    function weight_addr_l4 (
        neuron_idx : integer;
        input_idx  : integer
    ) return unsigned;

    -- Helper function: saturate int32 to int8
    function sat_int8 (
        x : int32_t
    ) return int8_t;

end package mnist_mlp_pkg;

package body mnist_mlp_pkg is
                                  
    -- Weight address mapping for each layer
    function weight_addr_l1 (
        neuron_idx : integer;
        input_idx  : integer
    ) return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L1_IN_SIZE + input_idx;
        return resize(to_unsigned(lin, W1_ADDR_WIDTH), W1_ADDR_WIDTH);
    end function;

    function weight_addr_l2 (
        neuron_idx : integer;
        input_idx  : integer
    ) return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L2_IN_SIZE + input_idx;
        return resize(to_unsigned(lin, W2_ADDR_WIDTH), W2_ADDR_WIDTH);
    end function;

    function weight_addr_l3 (
        neuron_idx : integer;
        input_idx  : integer
    ) return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L3_IN_SIZE + input_idx;
        return resize(to_unsigned(lin, W3_ADDR_WIDTH), W3_ADDR_WIDTH);
    end function;

    function weight_addr_l4 (
        neuron_idx : integer;
        input_idx  : integer
    ) return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L4_IN_SIZE + input_idx;
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

