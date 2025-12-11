library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mnist_mlp_pkg.all;

entity bias_rom is
    generic (
        INIT_FILE  : string;          -- "b1.mem"
        WORD_COUNT : integer;
        ADDR_WIDTH : integer
    );
    port (
        clk  : in  std_logic;
        addr : in  unsigned(ADDR_WIDTH-1 downto 0);
        dout : out int32_t
    );
end entity;

architecture rtl of bias_rom is

    -- Memory definition (signed 32-bit bias per entry)
    type mem_t is array (0 to WORD_COUNT-1) of int32_t;

    signal mem : mem_t := (others => (others => '0'));

    -- Attributes for BRAM + initialization
    attribute ram_style : string;
    attribute ram_style of mem : signal is "block";

    attribute ram_init_file : string;
    attribute ram_init_file of mem : signal is INIT_FILE;

    signal dout_reg : int32_t := (others => '0');

begin

    -- Synchronous read (BRAM)
    process(clk)
    begin
        if rising_edge(clk) then
            dout_reg <= mem(to_integer(addr));
        end if;
    end process;

    dout <= dout_reg;

end architecture;
