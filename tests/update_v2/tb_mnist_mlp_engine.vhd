library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use work.mnist_mlp_pkg.all;

-- requires w1.mem..w4.mem and b1.mem..b4.mem in the simulator run directory

entity tb_mnist_mlp_engine is end;

architecture sim of tb_mnist_mlp_engine is
  constant CLK_PERIOD:time:=10 ns;
  constant IMG_WORDS:integer:=L0_SIZE/4;
  type img_mem_t is array(0 to IMG_WORDS-1) of std_logic_vector(31 downto 0);
  signal clk:std_logic:='0';
  signal rst_n:std_logic:='0';
  signal start_i:std_logic:='0';
  signal busy_o,done_o,error_o:std_logic;
  signal img_addr_o:unsigned(15 downto 0);
  signal img_data_i:std_logic_vector(31 downto 0):=(others=>'0');
  signal logits_out:int8_vector_t(0 to L4_OUT_SIZE-1);
  signal img_mem:img_mem_t:=(others=>(others=>'0'));
  signal addr_q:unsigned(15 downto 0):=(others=>'0');
  function s8_to_int(x:int8_t)return integer is begin return to_integer(x);end;
begin
  clk<=not clk after CLK_PERIOD/2;
  dut:entity work.mnist_mlp_engine
    port map(clk=>clk,rst_n=>rst_n,start_i=>start_i,busy_o=>busy_o,done_o=>done_o,error_o=>error_o,img_addr_o=>img_addr_o,img_data_i=>img_data_i,logits_out=>logits_out);
  init_proc:process
    variable b:integer:=0;
    variable w:std_logic_vector(31 downto 0);
  begin
    for i in 0 to IMG_WORDS-1 loop
      w:=(others=>'0');
      for k in 0 to 3 loop
        w(8*k+7 downto 8*k):=std_logic_vector(to_signed((b mod 128),8));
        b:=b+1;
      end loop;
      img_mem(i)<=w;
    end loop;
    wait;
  end process;
  bram_proc:process(clk)
  begin
    if rising_edge(clk) then
      addr_q<=img_addr_o;
      if to_integer(addr_q)<IMG_WORDS then img_data_i<=img_mem(to_integer(addr_q));else img_data_i<=(others=>'0');end if;
    end if;
  end process;
  stim:process
  begin
    rst_n<='0';
    wait for 20*CLK_PERIOD;
    rst_n<='1';
    wait for 5*CLK_PERIOD;
    start_i<='1';
    wait for CLK_PERIOD;
    start_i<='0';
    wait until rising_edge(clk) and done_o='1';
    assert error_o='0' report "engine error asserted" severity failure;
    for i in 0 to L4_OUT_SIZE-1 loop
      report "logit["&integer'image(i)&"]="&integer'image(s8_to_int(logits_out(i)));
    end loop;
    stop(0);
  end process;
end architecture;
