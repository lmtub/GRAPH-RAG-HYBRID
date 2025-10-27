static void disas_system(DisasContext *s, uint32_t insn)

{

    unsigned int l, op0, op1, crn, crm, op2, rt;

    l = extract32(insn, 21, 1);

    op0 = extract32(insn, 19, 2);

    op1 = extract32(insn, 16, 3);

    crn = extract32(insn, 12, 4);

    crm = extract32(insn, 8, 4);

    op2 = extract32(insn, 5, 3);

    rt = extract32(insn, 0, 5);



    if (op0 == 0) {

        if (l || rt != 31) {

            unallocated_encoding(s);

            return;

        }

        switch (crn) {

        case 2: /* C5.6.68 HINT */

            handle_hint(s, insn, op1, op2, crm);

            break;

        case 3: /* CLREX, DSB, DMB, ISB */

            handle_sync(s, insn, op1, op2, crm);

            break;

        case 4: /* C5.6.130 MSR (immediate) */

            handle_msr_i(s, insn, op1, op2, crm);

            break;

        default:

            unallocated_encoding(s);

            break;

        }

        return;

    }



    if (op0 == 1) {

        /* C5.6.204 SYS */

        handle_sys(s, insn, l, op1, op2, crn, crm, rt);

    } else if (l) { /* op0 > 1 */

        /* C5.6.129 MRS - move from system register */

        handle_mrs(s, insn, op0, op1, op2, crn, crm, rt);

    } else {

        /* C5.6.131 MSR (register) - move to system register */

        handle_msr(s, insn, op0, op1, op2, crn, crm, rt);

    }

}
