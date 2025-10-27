static TileExcp decode_y0(DisasContext *dc, tilegx_bundle_bits bundle)

{

    unsigned opc = get_Opcode_Y0(bundle);

    unsigned ext = get_RRROpcodeExtension_Y0(bundle);

    unsigned dest = get_Dest_Y0(bundle);

    unsigned srca = get_SrcA_Y0(bundle);

    unsigned srcb;

    int imm;



    switch (opc) {

    case RRR_1_OPCODE_Y0:

        if (ext == UNARY_RRR_1_OPCODE_Y0) {

            ext = get_UnaryOpcodeExtension_Y0(bundle);

            return gen_rr_opcode(dc, OE(opc, ext, Y0), dest, srca);

        }

        /* fallthru */

    case RRR_0_OPCODE_Y0:

    case RRR_2_OPCODE_Y0:

    case RRR_3_OPCODE_Y0:

    case RRR_4_OPCODE_Y0:

    case RRR_5_OPCODE_Y0:

    case RRR_6_OPCODE_Y0:

    case RRR_7_OPCODE_Y0:

    case RRR_8_OPCODE_Y0:

    case RRR_9_OPCODE_Y0:

        srcb = get_SrcB_Y0(bundle);

        return gen_rrr_opcode(dc, OE(opc, ext, Y0), dest, srca, srcb);



    case SHIFT_OPCODE_Y0:

        ext = get_ShiftOpcodeExtension_Y0(bundle);

        imm = get_ShAmt_Y0(bundle);

        return gen_rri_opcode(dc, OE(opc, ext, Y0), dest, srca, imm);



    case ADDI_OPCODE_Y0:

    case ADDXI_OPCODE_Y0:

    case ANDI_OPCODE_Y0:

    case CMPEQI_OPCODE_Y0:

    case CMPLTSI_OPCODE_Y0:

        imm = (int8_t)get_Imm8_Y0(bundle);

        return gen_rri_opcode(dc, OE(opc, 0, Y0), dest, srca, imm);



    default:

        return TILEGX_EXCP_OPCODE_UNIMPLEMENTED;

    }

}
