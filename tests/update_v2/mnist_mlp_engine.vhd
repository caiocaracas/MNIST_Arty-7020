-- File: mnist_mlp_engine.vhd
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

        -- Input image (layer 0 activations)
        image_in   : in  int8_vector_t(0 to L0_SIZE-1);

        -- Output logits (layer 4 activations)
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
end architecture rtl;