library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

use work.mnist_mlp_pkg.all;

entity bias_rom is
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
end entity bias_rom;

architecture rtl of bias_rom is

    constant DATA_WIDTH : integer := 32;

    type ram_t is array (0 to WORD_COUNT-1) of std_logic_vector(DATA_WIDTH-1 downto 0);

    impure function init_ram_from_file (
        file_name : in string
    ) return ram_t is
        file     f        : text open read_mode is file_name;
        variable line_buf : line;
        variable tmp      : bit_vector(DATA_WIDTH-1 downto 0);
        variable ram_v    : ram_t := (others => (others => '0'));
    begin
        for i in ram_t'range loop
            exit when endfile(f);
            readline(f, line_buf);
            read(line_buf, tmp);
            ram_v(i) := to_stdlogicvector(tmp); 
        end loop;
        return ram_v;
    end function;

    signal mem : ram_t := init_ram_from_file(INIT_FILE);

    attribute ram_style : string;
    attribute ram_style of mem : signal is "block";

    signal dout_raw : std_logic_vector(DATA_WIDTH-1 downto 0);

begin

    process (clk)
    begin
        if rising_edge(clk) then
            dout_raw <= mem(to_integer(addr));
        end if;
    end process;

    dout <= signed(dout_raw);

end architecture rtl;
