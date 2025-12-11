library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package mnist_mlp_pkg is

    subtype int8_t   is signed(7 downto 0);
    subtype int16_t  is signed(15 downto 0);
    subtype int32_t  is signed(31 downto 0);
    subtype int64_t  is signed(63 downto 0); 

    type int8_vector_t   is array (natural range <>) of int8_t;
    type int32_vector_t  is array (natural range <>) of int32_t;

    constant N_PAR : integer := 8;

    constant L0_SIZE : integer := 784;
    constant L1_SIZE : integer := 256;
    constant L2_SIZE : integer := 128;
    constant L3_SIZE : integer := 64;
    constant L4_SIZE : integer := 10;

    constant L1_IN_SIZE  : integer := L0_SIZE;
    constant L1_OUT_SIZE : integer := L1_SIZE;

    constant L2_IN_SIZE  : integer := L1_SIZE;
    constant L2_OUT_SIZE : integer := L2_SIZE;

    constant L3_IN_SIZE  : integer := L2_SIZE;
    constant L3_OUT_SIZE : integer := L3_SIZE;

    constant L4_IN_SIZE  : integer := L3_SIZE;
    constant L4_OUT_SIZE : integer := L4_SIZE;

    constant L1_IN_BLOCKS : integer := L1_IN_SIZE / N_PAR;
    constant L2_IN_BLOCKS : integer := L2_IN_SIZE / N_PAR;
    constant L3_IN_BLOCKS : integer := L3_IN_SIZE / N_PAR;
    constant L4_IN_BLOCKS : integer := L4_IN_SIZE / N_PAR;

    constant W1_WORD_COUNT : integer := L1_OUT_SIZE * L1_IN_BLOCKS;
    constant W2_WORD_COUNT : integer := L2_OUT_SIZE * L2_IN_BLOCKS;
    constant W3_WORD_COUNT : integer := L3_OUT_SIZE * L3_IN_BLOCKS;
    constant W4_WORD_COUNT : integer := L4_OUT_SIZE * L4_IN_BLOCKS;

    constant B1_COUNT : integer := L1_OUT_SIZE;
    constant B2_COUNT : integer := L2_OUT_SIZE;
    constant B3_COUNT : integer := L3_OUT_SIZE;
    constant B4_COUNT : integer := L4_OUT_SIZE;

    function clog2(n : integer) return integer;

    constant W1_ADDR_WIDTH : integer := clog2(W1_WORD_COUNT);
    constant W2_ADDR_WIDTH : integer := clog2(W2_WORD_COUNT);
    constant W3_ADDR_WIDTH : integer := clog2(W3_WORD_COUNT);
    constant W4_ADDR_WIDTH : integer := clog2(W4_WORD_COUNT);

    constant B1_ADDR_WIDTH : integer := clog2(B1_COUNT);
    constant B2_ADDR_WIDTH : integer := clog2(B2_COUNT);
    constant B3_ADDR_WIDTH : integer := clog2(B3_COUNT);
    constant B4_ADDR_WIDTH : integer := clog2(B4_COUNT);

    type weight_word_t is array (0 to N_PAR-1) of int8_t;

    type layer_qparams_t is record
        M      : int32_t;
        shift  : integer;
        zp_out : int8_t;
    end record;

    type qparams_array_t is array (1 to 4) of layer_qparams_t;

    constant QPARAMS : qparams_array_t := (
    1 => (
        M      => to_signed(4033381, 32),
        shift  => 31,
        zp_out => to_signed(0, 8)
    ),
    2 => (
        M      => to_signed(2057957, 32),
        shift  => 31,
        zp_out => to_signed(0, 8)
    ),
    3 => (
        M      => to_signed(777513, 32),
        shift  => 30,
        zp_out => to_signed(0, 8)
    ),
    4 => (
        M      => to_signed(114421, 32),
        shift  => 27,
        zp_out => to_signed(0, 8)
    )
);

    function weight_addr_l1(neuron_idx : integer; block_idx : integer)
        return unsigned;
    function weight_addr_l2(neuron_idx : integer; block_idx : integer)
        return unsigned;
    function weight_addr_l3(neuron_idx : integer; block_idx : integer)
        return unsigned;
    function weight_addr_l4(neuron_idx : integer; block_idx : integer)
        return unsigned;

    function sat_int8(x : int32_t) return int8_t;

end package mnist_mlp_pkg;

package body mnist_mlp_pkg is

    function clog2(n : integer) return integer is
        variable v : integer := n - 1;
        variable r : integer := 0;
    begin
        while v > 0 loop
            v := v / 2;
            r := r + 1;
        end loop;
        return r;
    end function;

    function weight_addr_l1(neuron_idx : integer; block_idx : integer)
        return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L1_IN_BLOCKS + block_idx;
        return resize(to_unsigned(lin, W1_ADDR_WIDTH), W1_ADDR_WIDTH);
    end function;

    function weight_addr_l2(neuron_idx : integer; block_idx : integer)
        return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L2_IN_BLOCKS + block_idx;
        return resize(to_unsigned(lin, W2_ADDR_WIDTH), W2_ADDR_WIDTH);
    end function;

    function weight_addr_l3(neuron_idx : integer; block_idx : integer)
        return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L3_IN_BLOCKS + block_idx;
        return resize(to_unsigned(lin, W3_ADDR_WIDTH), W3_ADDR_WIDTH);
    end function;

    function weight_addr_l4(neuron_idx : integer; block_idx : integer)
        return unsigned is
        variable lin : integer;
    begin
        lin := neuron_idx * L4_IN_BLOCKS + block_idx;
        return resize(to_unsigned(lin, W4_ADDR_WIDTH), W4_ADDR_WIDTH);
    end function;

    function sat_int8(x : int32_t) return int8_t is
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
