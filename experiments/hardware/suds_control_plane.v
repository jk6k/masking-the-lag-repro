// SUDS scheduler-derived quality-budget control plane.
// This intentionally small RTL block captures the comparator/threshold/tier
// sideband path used by the P3 overhead model.

module suds_control_plane #(
    parameter integer SLACK_BITS = 12,
    parameter integer TIER_BITS = 2
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  valid_i,
    input  wire [SLACK_BITS-1:0] slack_i,
    input  wire [SLACK_BITS-1:0] tau_low_i,
    input  wire [SLACK_BITS-1:0] tau_high_i,
    output reg                   valid_o,
    output reg  [TIER_BITS-1:0]  tier_o
);

    localparam [TIER_BITS-1:0] TIER_KEEP    = 2'b00;
    localparam [TIER_BITS-1:0] TIER_DEGRADE = 2'b01;
    localparam [TIER_BITS-1:0] TIER_PRUNE   = 2'b10;

    reg [SLACK_BITS-1:0] tau_low_q;
    reg [SLACK_BITS-1:0] tau_high_q;
    reg [TIER_BITS-1:0]  tier_next;

    always @(*) begin
        if (slack_i <= tau_low_q) begin
            tier_next = TIER_KEEP;
        end else if (slack_i <= tau_high_q) begin
            tier_next = TIER_DEGRADE;
        end else begin
            tier_next = TIER_PRUNE;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tau_low_q <= {SLACK_BITS{1'b0}};
            tau_high_q <= {SLACK_BITS{1'b0}};
            valid_o <= 1'b0;
            tier_o <= TIER_KEEP;
        end else begin
            tau_low_q <= tau_low_i;
            tau_high_q <= tau_high_i;
            valid_o <= valid_i;
            if (valid_i) begin
                tier_o <= tier_next;
            end
        end
    end

endmodule
