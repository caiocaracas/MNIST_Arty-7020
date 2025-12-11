library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mnist_mlp_pkg.all;

entity weight_rom is
    generic (
        INIT_FILE  : string;                -- file like "w1.mem"
        WORD_COUNT : integer;
        ADDR_WIDTH : integer
    );
    port (
        clk  : in  std_logic;
        addr : in  unsigned(ADDR_WIDTH-1 downto 0);
        dout : out weight_word_t
    );
end entity;

architecture rtl of weight_rom is

    -- Memory declaration
    -- Each word contains N_PAR int8_t weights
    type mem_t is array (0 to WORD_COUNT-1) of weight_word_t;

    -- Vivado cannot initialize VHDL composite array directly unless
    -- we use an attribute for the whole signal.
    signal mem : mem_t := (others => (others => (others => '0')));

    -- Force block RAM + memory initialization file
    attribute ram_style : string;
    attribute ram_style of mem : signal is "block";

    attribute ram_init_file : string;
    attribute ram_init_file of mem : signal is INIT_FILE;

    signal dout_reg : weight_word_t := (others => (others => '0'));

begin

    -- Synchronous read, BRAM-style
    process(clk)
    begin
        if rising_edge(clk) then
            dout_reg <= mem(to_integer(addr));
        end if;
    end process;

    dout <= dout_reg;

end architecture;
