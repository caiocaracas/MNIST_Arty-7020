library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity MNIST_accel is
	generic (

		-- Parameters of Axi Slave Bus Interface S00_AXI
		C_S00_AXI_DATA_WIDTH	: integer	:= 32;
		C_S00_AXI_ADDR_WIDTH	: integer	:= 4;

		-- Parameters of Axi Slave Bus Interface S00_AXIS
		C_S00_AXIS_TDATA_WIDTH	: integer	:= 32;

		-- Parameters of Axi Master Bus Interface M00_AXIS
		C_M00_AXIS_TDATA_WIDTH	: integer	:= 32;
		C_M00_AXIS_START_COUNT	: integer	:= 32;

		-- Parameters of Axi Slave Bus Interface S_AXI_INTR
		C_S_AXI_INTR_DATA_WIDTH	: integer	:= 32;
		C_S_AXI_INTR_ADDR_WIDTH	: integer	:= 5;
		C_NUM_OF_INTR	: integer	:= 1;
		C_INTR_SENSITIVITY	: std_logic_vector	:= x"FFFFFFFF";
		C_INTR_ACTIVE_STATE	: std_logic_vector	:= x"FFFFFFFF";
		C_IRQ_SENSITIVITY	: integer	:= 1;
		C_IRQ_ACTIVE_STATE	: integer	:= 1
	);
	port (
		-- Ports of Axi Slave Bus Interface S00_AXI
		s00_axi_aclk	: in std_logic;
		s00_axi_aresetn	: in std_logic;
		s00_axi_awaddr	: in std_logic_vector(C_S00_AXI_ADDR_WIDTH-1 downto 0);
		s00_axi_awprot	: in std_logic_vector(2 downto 0);
		s00_axi_awvalid	: in std_logic;
		s00_axi_awready	: out std_logic;
		s00_axi_wdata	: in std_logic_vector(C_S00_AXI_DATA_WIDTH-1 downto 0);
		s00_axi_wstrb	: in std_logic_vector((C_S00_AXI_DATA_WIDTH/8)-1 downto 0);
		s00_axi_wvalid	: in std_logic;
		s00_axi_wready	: out std_logic;
		s00_axi_bresp	: out std_logic_vector(1 downto 0);
		s00_axi_bvalid	: out std_logic;
		s00_axi_bready	: in std_logic;
		s00_axi_araddr	: in std_logic_vector(C_S00_AXI_ADDR_WIDTH-1 downto 0);
		s00_axi_arprot	: in std_logic_vector(2 downto 0);
		s00_axi_arvalid	: in std_logic;
		s00_axi_arready	: out std_logic;
		s00_axi_rdata	: out std_logic_vector(C_S00_AXI_DATA_WIDTH-1 downto 0);
		s00_axi_rresp	: out std_logic_vector(1 downto 0);
		s00_axi_rvalid	: out std_logic;
		s00_axi_rready	: in std_logic;

		-- Ports of Axi Slave Bus Interface S00_AXIS
		s00_axis_aclk	: in std_logic;
		s00_axis_aresetn	: in std_logic;
		s00_axis_tready	: out std_logic;
		s00_axis_tdata	: in std_logic_vector(C_S00_AXIS_TDATA_WIDTH-1 downto 0);
		s00_axis_tstrb	: in std_logic_vector((C_S00_AXIS_TDATA_WIDTH/8)-1 downto 0);
		s00_axis_tlast	: in std_logic;
		s00_axis_tvalid	: in std_logic;

		-- Ports of Axi Master Bus Interface M00_AXIS
		m00_axis_aclk	: in std_logic;
		m00_axis_aresetn	: in std_logic;
		m00_axis_tvalid	: out std_logic;
		m00_axis_tdata	: out std_logic_vector(C_M00_AXIS_TDATA_WIDTH-1 downto 0);
		m00_axis_tstrb	: out std_logic_vector((C_M00_AXIS_TDATA_WIDTH/8)-1 downto 0);
		m00_axis_tlast	: out std_logic;
		m00_axis_tready	: in std_logic;

		-- Ports of Axi Slave Bus Interface S_AXI_INTR
		s_axi_intr_aclk	: in std_logic;
		s_axi_intr_aresetn	: in std_logic;
		s_axi_intr_awaddr	: in std_logic_vector(C_S_AXI_INTR_ADDR_WIDTH-1 downto 0);
		s_axi_intr_awprot	: in std_logic_vector(2 downto 0);
		s_axi_intr_awvalid	: in std_logic;
		s_axi_intr_awready	: out std_logic;
		s_axi_intr_wdata	: in std_logic_vector(C_S_AXI_INTR_DATA_WIDTH-1 downto 0);
		s_axi_intr_wstrb	: in std_logic_vector((C_S_AXI_INTR_DATA_WIDTH/8)-1 downto 0);
		s_axi_intr_wvalid	: in std_logic;
		s_axi_intr_wready	: out std_logic;
		s_axi_intr_bresp	: out std_logic_vector(1 downto 0);
		s_axi_intr_bvalid	: out std_logic;
		s_axi_intr_bready	: in std_logic;
		s_axi_intr_araddr	: in std_logic_vector(C_S_AXI_INTR_ADDR_WIDTH-1 downto 0);
		s_axi_intr_arprot	: in std_logic_vector(2 downto 0);
		s_axi_intr_arvalid	: in std_logic;
		s_axi_intr_arready	: out std_logic;
		s_axi_intr_rdata	: out std_logic_vector(C_S_AXI_INTR_DATA_WIDTH-1 downto 0);
		s_axi_intr_rresp	: out std_logic_vector(1 downto 0);
		s_axi_intr_rvalid	: out std_logic;
		s_axi_intr_rready	: in std_logic;
		irq	: out std_logic
	);
end MNIST_accel;

architecture arch_imp of MNIST_accel is

  -- Component declarations
  -- AXI4-Lite control interface
  component MNIST_accel_slave_lite_v1_0_S00_AXI is
    generic (
      C_S_AXI_DATA_WIDTH : integer := 32;
      C_S_AXI_ADDR_WIDTH : integer := 4
    );
    port (
      -- User-side control/status interface
      ctrl_start_pulse : out std_logic;
      ctrl_irq_en      : out std_logic;
      status_busy      : in  std_logic;
      status_done      : in  std_logic;
      status_error     : in  std_logic := '0';
      img_length       : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);

      -- AXI4-Lite interface 
      S_AXI_ACLK    : in  std_logic;
      S_AXI_ARESETN : in  std_logic;
      S_AXI_AWADDR  : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
      S_AXI_AWPROT  : in  std_logic_vector(2 downto 0);
      S_AXI_AWVALID : in  std_logic;
      S_AXI_AWREADY : out std_logic;
      S_AXI_WDATA   : in  std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
      S_AXI_WSTRB   : in  std_logic_vector((C_S_AXI_DATA_WIDTH/8)-1 downto 0);
      S_AXI_WVALID  : in  std_logic;
      S_AXI_WREADY  : out std_logic;
      S_AXI_BRESP   : out std_logic_vector(1 downto 0);
      S_AXI_BVALID  : out std_logic;
      S_AXI_BREADY  : in  std_logic;
      S_AXI_ARADDR  : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
      S_AXI_ARPROT  : in  std_logic_vector(2 downto 0);
      S_AXI_ARVALID : in  std_logic;
      S_AXI_ARREADY : out std_logic;
      S_AXI_RDATA   : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
      S_AXI_RRESP   : out std_logic_vector(1 downto 0);
      S_AXI_RVALID  : out std_logic;
      S_AXI_RREADY  : in  std_logic
    );
  end component;

  -- AXI4-Stream slave (image ingestion)
  component MNIST_accel_slave_stream_v1_0_S00_AXIS is
    generic (
      C_S_AXIS_TDATA_WIDTH : integer := 32
    );
    port (
      S_AXIS_ACLK    : in  std_logic;
      S_AXIS_ARESETN : in  std_logic;
      S_AXIS_TREADY  : out std_logic;
      S_AXIS_TDATA   : in  std_logic_vector(C_S_AXIS_TDATA_WIDTH-1 downto 0);
      S_AXIS_TSTRB   : in  std_logic_vector((C_S_AXIS_TDATA_WIDTH/8)-1 downto 0);
      S_AXIS_TLAST   : in  std_logic;
      S_AXIS_TVALID  : in  std_logic;

      img_length_bytes : in  std_logic_vector(31 downto 0);
      img_word_wr_en   : out std_logic;
      img_word_wr_addr : out unsigned(15 downto 0);
      img_word_wr_data : out std_logic_vector(C_S_AXIS_TDATA_WIDTH-1 downto 0);
      img_done         : out std_logic;
      clear_img_done   : in  std_logic
    );
  end component;

  -- AXI4-Stream master (logit emission)
  component MNIST_accel_master_stream_v1_0_M00_AXIS is
    generic (
      C_M_AXIS_TDATA_WIDTH : integer := 32;
      C_M_AXIS_START_COUNT : integer := 32
    );
    port (
      M_AXIS_ACLK    : in  std_logic;
      M_AXIS_ARESETN : in  std_logic;
      M_AXIS_TVALID  : out std_logic;
      M_AXIS_TDATA   : out std_logic_vector(C_M_AXIS_TDATA_WIDTH-1 downto 0);
      M_AXIS_TSTRB   : out std_logic_vector((C_M_AXIS_TDATA_WIDTH/8)-1 downto 0);
      M_AXIS_TLAST   : out std_logic;
      M_AXIS_TREADY  : in  std_logic;

      logits_data  : in  std_logic_vector(79 downto 0);
      logits_valid : in  std_logic;
      logits_sent  : out std_logic
    );
  end component;

  -- Interrupt controller
  component MNIST_accel_slave_lite_inter_v1_0_S_AXI_INTR is
    generic (
      C_S_AXI_DATA_WIDTH  : integer := 32;
      C_S_AXI_ADDR_WIDTH  : integer := 5;
      C_NUM_OF_INTR       : integer := 1;
      C_INTR_SENSITIVITY  : std_logic_vector := x"FFFFFFFF";
      C_INTR_ACTIVE_STATE : std_logic_vector := x"FFFFFFFF";
      C_IRQ_SENSITIVITY   : integer := 1;
      C_IRQ_ACTIVE_STATE  : integer := 1
    );
    port (
      S_AXI_ACLK    : in  std_logic;
      S_AXI_ARESETN : in  std_logic;
      S_AXI_AWADDR  : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
      S_AXI_AWPROT  : in  std_logic_vector(2 downto 0);
      S_AXI_AWVALID : in  std_logic;
      S_AXI_AWREADY : out std_logic;
      S_AXI_WDATA   : in  std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
      S_AXI_WSTRB   : in  std_logic_vector((C_S_AXI_DATA_WIDTH/8)-1 downto 0);
      S_AXI_WVALID  : in  std_logic;
      S_AXI_WREADY  : out std_logic;
      S_AXI_BRESP   : out std_logic_vector(1 downto 0);
      S_AXI_BVALID  : out std_logic;
      S_AXI_BREADY  : in  std_logic;
      S_AXI_ARADDR  : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
      S_AXI_ARPROT  : in  std_logic_vector(2 downto 0);
      S_AXI_ARVALID : in  std_logic;
      S_AXI_ARREADY : out std_logic;
      S_AXI_RDATA   : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
      S_AXI_RRESP   : out std_logic_vector(1 downto 0);
      S_AXI_RVALID  : out std_logic;
      S_AXI_RREADY  : in  std_logic;
      irq           : out std_logic
    );
  end component;

  -- Internal control-path signals
  signal ctrl_start_pulse : std_logic;
  signal ctrl_irq_en      : std_logic;

  signal status_busy  : std_logic := '0';
  signal status_done  : std_logic := '0';
  signal status_error : std_logic := '0';

  signal img_length_reg : std_logic_vector(31 downto 0);

  -- Image ingestion interface signals
  signal img_word_wr_en   : std_logic;
  signal img_word_wr_addr : unsigned(15 downto 0);
  signal img_word_wr_data : std_logic_vector(C_S00_AXIS_TDATA_WIDTH-1 downto 0);
  signal img_done         : std_logic;
  signal clear_img_done   : std_logic := '0';

  -- Logit output interface signals
  signal logits_data_vec : std_logic_vector(79 downto 0) := (others => '0');
  signal logits_valid    : std_logic := '0';
  signal logits_sent     : std_logic;

  -- High-level accelerator finite state machine
  type accel_state_t is (IDLE, WAIT_INPUT, COMPUTE, SEND_OUTPUT);
  signal accel_state : accel_state_t := IDLE;

begin

  -- AXI4-Lite control interface instance
  MNIST_accel_slave_lite_v1_0_S00_AXI_inst : MNIST_accel_slave_lite_v1_0_S00_AXI
    generic map (
      C_S_AXI_DATA_WIDTH => C_S00_AXI_DATA_WIDTH,
      C_S_AXI_ADDR_WIDTH => C_S00_AXI_ADDR_WIDTH
    )
    port map (
      ctrl_start_pulse => ctrl_start_pulse,
      ctrl_irq_en      => ctrl_irq_en,
      status_busy      => status_busy,
      status_done      => status_done,
      status_error     => status_error,
      img_length       => img_length_reg,
      -- AXI-Lite passthrough
      S_AXI_ACLK    => s00_axi_aclk,
      S_AXI_ARESETN => s00_axi_aresetn,
      S_AXI_AWADDR  => s00_axi_awaddr,
      S_AXI_AWPROT  => s00_axi_awprot,
      S_AXI_AWVALID => s00_axi_awvalid,
      S_AXI_AWREADY => s00_axi_awready,
      S_AXI_WDATA   => s00_axi_wdata,
      S_AXI_WSTRB   => s00_axi_wstrb,
      S_AXI_WVALID  => s00_axi_wvalid,
      S_AXI_WREADY  => s00_axi_wready,
      S_AXI_BRESP   => s00_axi_bresp,
      S_AXI_BVALID  => s00_axi_bvalid,
      S_AXI_BREADY  => s00_axi_bready,
      S_AXI_ARADDR  => s00_axi_araddr,
      S_AXI_ARPROT  => s00_axi_arprot,
      S_AXI_ARVALID => s00_axi_arvalid,
      S_AXI_ARREADY => s00_axi_arready,
      S_AXI_RDATA   => s00_axi_rdata,
      S_AXI_RRESP   => s00_axi_rresp,
      S_AXI_RVALID  => s00_axi_rvalid,
      S_AXI_RREADY  => s00_axi_rready
    );

  -- AXI4-Stream slave (image ingestion)
  MNIST_accel_slave_stream_v1_0_S00_AXIS_inst : MNIST_accel_slave_stream_v1_0_S00_AXIS
    generic map (
      C_S_AXIS_TDATA_WIDTH => C_S00_AXIS_TDATA_WIDTH
    )
    port map (
      S_AXIS_ACLK      => s00_axis_aclk,
      S_AXIS_ARESETN   => s00_axis_aresetn,
      S_AXIS_TREADY    => s00_axis_tready,
      S_AXIS_TDATA     => s00_axis_tdata,
      S_AXIS_TSTRB     => s00_axis_tstrb,
      S_AXIS_TLAST     => s00_axis_tlast,
      S_AXIS_TVALID    => s00_axis_tvalid,

      img_length_bytes => img_length_reg,
      img_word_wr_en   => img_word_wr_en,
      img_word_wr_addr => img_word_wr_addr,
      img_word_wr_data => img_word_wr_data,
      img_done         => img_done,
      clear_img_done   => clear_img_done
    );

  -- AXI4-Stream master (logit emission)
  MNIST_accel_master_stream_v1_0_M00_AXIS_inst :
    MNIST_accel_master_stream_v1_0_M00_AXIS
    generic map (
      C_M_AXIS_TDATA_WIDTH => C_M00_AXIS_TDATA_WIDTH,
      C_M_AXIS_START_COUNT => C_M00_AXIS_START_COUNT
    )
    port map (
      M_AXIS_ACLK    => m00_axis_aclk,
      M_AXIS_ARESETN => m00_axis_aresetn,
      M_AXIS_TVALID  => m00_axis_tvalid,
      M_AXIS_TDATA   => m00_axis_tdata,
      M_AXIS_TSTRB   => m00_axis_tstrb,
      M_AXIS_TLAST   => m00_axis_tlast,
      M_AXIS_TREADY  => m00_axis_tready,

      logits_data  => logits_data_vec,
      logits_valid => logits_valid,
      logits_sent  => logits_sent
    );
    
  -- Interrupt controller instance 
  MNIST_accel_slave_lite_inter_v1_0_S_AXI_INTR_inst :
    MNIST_accel_slave_lite_inter_v1_0_S_AXI_INTR
    generic map (
      C_S_AXI_DATA_WIDTH  => C_S_AXI_INTR_DATA_WIDTH,
      C_S_AXI_ADDR_WIDTH  => C_S_AXI_INTR_ADDR_WIDTH,
      C_NUM_OF_INTR       => C_NUM_OF_INTR,
      C_INTR_SENSITIVITY  => C_INTR_SENSITIVITY,
      C_INTR_ACTIVE_STATE => C_INTR_ACTIVE_STATE,
      C_IRQ_SENSITIVITY   => C_IRQ_SENSITIVITY,
      C_IRQ_ACTIVE_STATE  => C_IRQ_ACTIVE_STATE
    )
    port map (
      S_AXI_ACLK    => s_axi_intr_aclk,
      S_AXI_ARESETN => s_axi_intr_aresetn,
      S_AXI_AWADDR  => s_axi_intr_awaddr,
      S_AXI_AWPROT  => s_axi_intr_awprot,
      S_AXI_AWVALID => s_axi_intr_awvalid,
      S_AXI_AWREADY => s_axi_intr_awready,
      S_AXI_WDATA   => s_axi_intr_wdata,
      S_AXI_WSTRB   => s_axi_intr_wstrb,
      S_AXI_WVALID  => s_axi_intr_wvalid,
      S_AXI_WREADY  => s_axi_intr_wready,
      S_AXI_BRESP   => s_axi_intr_bresp,
      S_AXI_BVALID  => s_axi_intr_bvalid,
      S_AXI_BREADY  => s_axi_intr_bready,
      S_AXI_ARADDR  => s_axi_intr_araddr,
      S_AXI_ARPROT  => s_axi_intr_arprot,
      S_AXI_ARVALID => s_axi_intr_arvalid,
      S_AXI_ARREADY => s_axi_intr_arready,
      S_AXI_RDATA   => s_axi_intr_rdata,
      S_AXI_RRESP   => s_axi_intr_rresp,
      S_AXI_RVALID  => s_axi_intr_rvalid,
      S_AXI_RREADY  => s_axi_intr_rready,
      irq           => irq
    );

  -- High-level accelerator state machine
  fsm_proc : process (s00_axi_aclk)
  begin
    if rising_edge(s00_axi_aclk) then
      if s00_axi_aresetn = '0' then
        accel_state    <= IDLE;
        status_busy    <= '0';
        status_done    <= '0';
        status_error   <= '0';
        clear_img_done <= '0';
        logits_valid   <= '0';
      else
        clear_img_done <= '0';
        logits_valid   <= logits_valid;

        case accel_state is
          when IDLE =>
            status_busy <= '0';

            if ctrl_start_pulse = '1' then
              status_done <= '0';
              status_busy <= '1';
              accel_state <= WAIT_INPUT;
            end if;

          when WAIT_INPUT =>
            if img_done = '1' then
              clear_img_done <= '1';
              accel_state    <= COMPUTE;
            end if;

          when COMPUTE =>
            -- Placeholder: inference core
            logits_data_vec <= (others => '0'); 
            logits_valid    <= '1';
            accel_state     <= SEND_OUTPUT;

          when SEND_OUTPUT =>
            if logits_sent = '1' then
              logits_valid <= '0';
              status_busy  <= '0';
              status_done  <= '1';
              accel_state  <= IDLE;
            end if;
        end case;
      end if;
    end if;
  end process fsm_proc;
end architecture arch_imp;
