// SUDS scheduler-derived quality-budget control plane.
//
// This block is intentionally compact, but it is no longer only a tier
// comparator.  It materializes the control path used by the TETC R7 evidence:
// budget/configuration registers, a scheduler-slack tier decision, a sideband
// command encoder, a tile-command handshake, and a small issue state machine.
// It is synthesizable evidence for architecture-level control accounting, not
// a physical-design or timing-closure claim.

module suds_control_plane #(
    parameter integer SLACK_BITS = 12,
    parameter integer TIER_BITS = 2,
    parameter integer BUDGET_BITS = 16,
    parameter integer GROUP_BITS = 8,
    parameter integer KERNEL_ID_BITS = 8,
    parameter integer QUEUE_BITS = 6,
    parameter integer CMD_BITS = 32
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  cfg_valid_i,
    input  wire [3:0]            cfg_addr_i,
    input  wire [BUDGET_BITS-1:0] cfg_data_i,
    output wire                  cfg_ready_o,
    input  wire                  valid_i,
    output wire                  ready_o,
    input  wire [KERNEL_ID_BITS-1:0] kernel_id_i,
    input  wire [SLACK_BITS-1:0] slack_i,
    input  wire [SLACK_BITS-1:0] deadline_delta_i,
    input  wire [QUEUE_BITS-1:0] queue_depth_i,
    input  wire [7:0]            selector_score_i,
    input  wire                  tile_ready_i,
    output reg                   valid_o,
    output reg  [TIER_BITS-1:0]  tier_o,
    output reg  [CMD_BITS-1:0]   tile_cmd_o,
    output reg                   overflow_o,
    output reg  [1:0]            state_o
);

    localparam [TIER_BITS-1:0] TIER_KEEP    = 2'b00;
    localparam [TIER_BITS-1:0] TIER_DEGRADE = 2'b01;
    localparam [TIER_BITS-1:0] TIER_PRUNE   = 2'b10;

    localparam [1:0] STATE_IDLE   = 2'b00;
    localparam [1:0] STATE_ENCODE = 2'b01;
    localparam [1:0] STATE_ISSUE  = 2'b10;
    localparam [1:0] STATE_WAIT   = 2'b11;

    reg [SLACK_BITS-1:0] tau_low_q;
    reg [SLACK_BITS-1:0] tau_high_q;
    reg [BUDGET_BITS-1:0] keep_budget_q;
    reg [BUDGET_BITS-1:0] degrade_budget_q;
    reg [BUDGET_BITS-1:0] prune_budget_q;
    reg [GROUP_BITS-1:0] sideband_group_q;
    reg [QUEUE_BITS-1:0] queue_limit_q;
    reg [7:0] score_guard_q;

    reg [KERNEL_ID_BITS-1:0] kernel_id_q;
    reg [SLACK_BITS-1:0] slack_q;
    reg [SLACK_BITS-1:0] deadline_delta_q;
    reg [QUEUE_BITS-1:0] queue_depth_q;
    reg [7:0] selector_score_q;

    reg [TIER_BITS-1:0]  tier_next;
    reg                  overflow_next;
    reg [CMD_BITS-1:0]   command_next;

    wire keep_budget_empty = (keep_budget_q == {BUDGET_BITS{1'b0}});
    wire degrade_budget_empty = (degrade_budget_q == {BUDGET_BITS{1'b0}});
    wire prune_budget_empty = (prune_budget_q == {BUDGET_BITS{1'b0}});
    wire queue_pressure = (queue_depth_q >= queue_limit_q);
    wire score_guard_hit = (selector_score_q >= score_guard_q);

    assign cfg_ready_o = 1'b1;
    assign ready_o = (state_o == STATE_IDLE);

    always @(*) begin
        overflow_next = 1'b0;
        if ((slack_q <= tau_low_q) || score_guard_hit) begin
            tier_next = TIER_KEEP;
            overflow_next = keep_budget_empty;
        end else if ((slack_q <= tau_high_q) && !degrade_budget_empty) begin
            tier_next = TIER_DEGRADE;
        end else if (!prune_budget_empty || queue_pressure) begin
            tier_next = TIER_PRUNE;
        end else begin
            tier_next = TIER_DEGRADE;
            overflow_next = 1'b1;
        end
    end

    always @(*) begin
        command_next = {CMD_BITS{1'b0}};
        command_next[1:0] = tier_next;
        command_next[9:2] = kernel_id_q[7:0];
        command_next[17:10] = sideband_group_q[7:0];
        command_next[29:18] = deadline_delta_q[11:0];
        command_next[30] = overflow_next;
        command_next[31] = queue_pressure;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tau_low_q <= {{(SLACK_BITS-4){1'b0}}, 4'd3};
            tau_high_q <= {{(SLACK_BITS-4){1'b0}}, 4'd9};
            keep_budget_q <= {BUDGET_BITS{1'b1}};
            degrade_budget_q <= {BUDGET_BITS{1'b1}};
            prune_budget_q <= {BUDGET_BITS{1'b1}};
            sideband_group_q <= {{(GROUP_BITS-6){1'b0}}, 6'd32};
            queue_limit_q <= {{(QUEUE_BITS-3){1'b0}}, 3'd4};
            score_guard_q <= 8'd224;
            kernel_id_q <= {KERNEL_ID_BITS{1'b0}};
            slack_q <= {SLACK_BITS{1'b0}};
            deadline_delta_q <= {SLACK_BITS{1'b0}};
            queue_depth_q <= {QUEUE_BITS{1'b0}};
            selector_score_q <= 8'd0;
            valid_o <= 1'b0;
            tier_o <= TIER_KEEP;
            tile_cmd_o <= {CMD_BITS{1'b0}};
            overflow_o <= 1'b0;
            state_o <= STATE_IDLE;
        end else begin
            if (cfg_valid_i) begin
                case (cfg_addr_i)
                    4'd0: tau_low_q <= cfg_data_i[SLACK_BITS-1:0];
                    4'd1: tau_high_q <= cfg_data_i[SLACK_BITS-1:0];
                    4'd2: keep_budget_q <= cfg_data_i;
                    4'd3: degrade_budget_q <= cfg_data_i;
                    4'd4: prune_budget_q <= cfg_data_i;
                    4'd5: sideband_group_q <= cfg_data_i[GROUP_BITS-1:0];
                    4'd6: queue_limit_q <= cfg_data_i[QUEUE_BITS-1:0];
                    4'd7: score_guard_q <= cfg_data_i[7:0];
                    default: begin
                        tau_low_q <= tau_low_q;
                    end
                endcase
            end

            case (state_o)
                STATE_IDLE: begin
                    valid_o <= 1'b0;
                    overflow_o <= 1'b0;
                    if (valid_i) begin
                        kernel_id_q <= kernel_id_i;
                        slack_q <= slack_i;
                        deadline_delta_q <= deadline_delta_i;
                        queue_depth_q <= queue_depth_i;
                        selector_score_q <= selector_score_i;
                        state_o <= STATE_ENCODE;
                    end
                end
                STATE_ENCODE: begin
                    tier_o <= tier_next;
                    tile_cmd_o <= command_next;
                    overflow_o <= overflow_next;
                    state_o <= STATE_ISSUE;
                end
                STATE_ISSUE: begin
                    valid_o <= 1'b1;
                    if (tile_ready_i) begin
                        case (tier_o)
                            TIER_KEEP: begin
                                if (!keep_budget_empty) begin
                                    keep_budget_q <= keep_budget_q - {{(BUDGET_BITS-1){1'b0}}, 1'b1};
                                end
                            end
                            TIER_DEGRADE: begin
                                if (!degrade_budget_empty) begin
                                    degrade_budget_q <= degrade_budget_q - {{(BUDGET_BITS-1){1'b0}}, 1'b1};
                                end
                            end
                            default: begin
                                if (!prune_budget_empty) begin
                                    prune_budget_q <= prune_budget_q - {{(BUDGET_BITS-1){1'b0}}, 1'b1};
                                end
                            end
                        endcase
                        valid_o <= 1'b0;
                        state_o <= STATE_IDLE;
                    end else begin
                        state_o <= STATE_WAIT;
                    end
                end
                default: begin
                    valid_o <= 1'b1;
                    if (tile_ready_i) begin
                        valid_o <= 1'b0;
                        state_o <= STATE_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
