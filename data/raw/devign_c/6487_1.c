static void lsi_execute_script(LSIState *s)

{

    uint32_t insn;

    uint32_t addr;

    int opcode;



    s->istat1 |= LSI_ISTAT1_SRUN;

again:

    insn = read_dword(s, s->dsp);

    addr = read_dword(s, s->dsp + 4);

    DPRINTF("SCRIPTS dsp=%08x opcode %08x arg %08x\n", s->dsp, insn, addr);

    s->dsps = addr;

    s->dcmd = insn >> 24;

    s->dsp += 8;

    switch (insn >> 30) {

    case 0: /* Block move.  */

        if (s->sist1 & LSI_SIST1_STO) {

            DPRINTF("Delayed select timeout\n");

            lsi_stop_script(s);

            break;

        }

        s->dbc = insn & 0xffffff;

        s->rbc = s->dbc;

        if (insn & (1 << 29)) {

            /* Indirect addressing.  */

            addr = read_dword(s, addr);

        } else if (insn & (1 << 28)) {

            uint32_t buf[2];

            int32_t offset;

            /* Table indirect addressing.  */

            offset = sxt24(addr);

            cpu_physical_memory_read(s->dsa + offset, (uint8_t *)buf, 8);

            s->dbc = cpu_to_le32(buf[0]);

            s->rbc = s->dbc;

            addr = cpu_to_le32(buf[1]);

        }

        if ((s->sstat1 & PHASE_MASK) != ((insn >> 24) & 7)) {

            DPRINTF("Wrong phase got %d expected %d\n",

                    s->sstat1 & PHASE_MASK, (insn >> 24) & 7);

            lsi_script_scsi_interrupt(s, LSI_SIST0_MA, 0);

            break;

        }

        s->dnad = addr;

        /* ??? Set ESA.  */

        s->ia = s->dsp - 8;

        switch (s->sstat1 & 0x7) {

        case PHASE_DO:

            s->waiting = 2;

            lsi_do_dma(s, 1);

            if (s->waiting)

                s->waiting = 3;

            break;

        case PHASE_DI:

            s->waiting = 2;

            lsi_do_dma(s, 0);

            if (s->waiting)

                s->waiting = 3;

            break;

        case PHASE_CMD:

            lsi_do_command(s);

            break;

        case PHASE_ST:

            lsi_do_status(s);

            break;

        case PHASE_MO:

            lsi_do_msgout(s);

            break;

        case PHASE_MI:

            lsi_do_msgin(s);

            break;

        default:

            BADF("Unimplemented phase %d\n", s->sstat1 & PHASE_MASK);

            exit(1);

        }

        s->dfifo = s->dbc & 0xff;

        s->ctest5 = (s->ctest5 & 0xfc) | ((s->dbc >> 8) & 3);

        s->sbc = s->dbc;

        s->rbc -= s->dbc;

        s->ua = addr + s->dbc;

        break;



    case 1: /* IO or Read/Write instruction.  */

        opcode = (insn >> 27) & 7;

        if (opcode < 5) {

            uint32_t id;



            if (insn & (1 << 25)) {

                id = read_dword(s, s->dsa + sxt24(insn));

            } else {

                id = addr;

            }

            id = (id >> 16) & 0xf;

            if (insn & (1 << 26)) {

                addr = s->dsp + sxt24(addr);

            }

            s->dnad = addr;

            switch (opcode) {

            case 0: /* Select */

                s->sdid = id;

                if (s->current_dma_len && (s->ssid & 0xf) == id) {

                    DPRINTF("Already reselected by target %d\n", id);

                    break;

                }

                s->sstat0 |= LSI_SSTAT0_WOA;

                s->scntl1 &= ~LSI_SCNTL1_IARB;

                if (id >= LSI_MAX_DEVS || !s->scsi_dev[id]) {

                    DPRINTF("Selected absent target %d\n", id);

                    lsi_script_scsi_interrupt(s, 0, LSI_SIST1_STO);

                    lsi_disconnect(s);

                    break;

                }

                DPRINTF("Selected target %d%s\n",

                        id, insn & (1 << 3) ? " ATN" : "");

                /* ??? Linux drivers compain when this is set.  Maybe

                   it only applies in low-level mode (unimplemented).

                lsi_script_scsi_interrupt(s, LSI_SIST0_CMP, 0); */

                s->current_dev = s->scsi_dev[id];

                s->current_tag = id << 8;

                s->scntl1 |= LSI_SCNTL1_CON;

                if (insn & (1 << 3)) {

                    s->socl |= LSI_SOCL_ATN;

                }

                lsi_set_phase(s, PHASE_MO);

                break;

            case 1: /* Disconnect */

                DPRINTF("Wait Disconect\n");

                s->scntl1 &= ~LSI_SCNTL1_CON;

                break;

            case 2: /* Wait Reselect */

                lsi_wait_reselect(s);

                break;

            case 3: /* Set */

                DPRINTF("Set%s%s%s%s\n",

                        insn & (1 << 3) ? " ATN" : "",

                        insn & (1 << 6) ? " ACK" : "",

                        insn & (1 << 9) ? " TM" : "",

                        insn & (1 << 10) ? " CC" : "");

                if (insn & (1 << 3)) {

                    s->socl |= LSI_SOCL_ATN;

                    lsi_set_phase(s, PHASE_MO);

                }

                if (insn & (1 << 9)) {

                    BADF("Target mode not implemented\n");

                    exit(1);

                }

                if (insn & (1 << 10))

                    s->carry = 1;

                break;

            case 4: /* Clear */

                DPRINTF("Clear%s%s%s%s\n",

                        insn & (1 << 3) ? " ATN" : "",

                        insn & (1 << 6) ? " ACK" : "",

                        insn & (1 << 9) ? " TM" : "",

                        insn & (1 << 10) ? " CC" : "");

                if (insn & (1 << 3)) {

                    s->socl &= ~LSI_SOCL_ATN;

                }

                if (insn & (1 << 10))

                    s->carry = 0;

                break;

            }

        } else {

            uint8_t op0;

            uint8_t op1;

            uint8_t data8;

            int reg;

            int operator;

#ifdef DEBUG_LSI

            static const char *opcode_names[3] =

                {"Write", "Read", "Read-Modify-Write"};

            static const char *operator_names[8] =

                {"MOV", "SHL", "OR", "XOR", "AND", "SHR", "ADD", "ADC"};

#endif



            reg = ((insn >> 16) & 0x7f) | (insn & 0x80);

            data8 = (insn >> 8) & 0xff;

            opcode = (insn >> 27) & 7;

            operator = (insn >> 24) & 7;

            DPRINTF("%s reg 0x%x %s data8=0x%02x sfbr=0x%02x%s\n",

                    opcode_names[opcode - 5], reg,

                    operator_names[operator], data8, s->sfbr,

                    (insn & (1 << 23)) ? " SFBR" : "");

            op0 = op1 = 0;

            switch (opcode) {

            case 5: /* From SFBR */

                op0 = s->sfbr;

                op1 = data8;

                break;

            case 6: /* To SFBR */

                if (operator)

                    op0 = lsi_reg_readb(s, reg);

                op1 = data8;

                break;

            case 7: /* Read-modify-write */

                if (operator)

                    op0 = lsi_reg_readb(s, reg);

                if (insn & (1 << 23)) {

                    op1 = s->sfbr;

                } else {

                    op1 = data8;

                }

                break;

            }



            switch (operator) {

            case 0: /* move */

                op0 = op1;

                break;

            case 1: /* Shift left */

                op1 = op0 >> 7;

                op0 = (op0 << 1) | s->carry;

                s->carry = op1;

                break;

            case 2: /* OR */

                op0 |= op1;

                break;

            case 3: /* XOR */

                op0 ^= op1;

                break;

            case 4: /* AND */

                op0 &= op1;

                break;

            case 5: /* SHR */

                op1 = op0 & 1;

                op0 = (op0 >> 1) | (s->carry << 7);

                s->carry = op1;

                break;

            case 6: /* ADD */

                op0 += op1;

                s->carry = op0 < op1;

                break;

            case 7: /* ADC */

                op0 += op1 + s->carry;

                if (s->carry)

                    s->carry = op0 <= op1;

                else

                    s->carry = op0 < op1;

                break;

            }



            switch (opcode) {

            case 5: /* From SFBR */

            case 7: /* Read-modify-write */

                lsi_reg_writeb(s, reg, op0);

                break;

            case 6: /* To SFBR */

                s->sfbr = op0;

                break;

            }

        }

        break;



    case 2: /* Transfer Control.  */

        {

            int cond;

            int jmp;



            if ((insn & 0x002e0000) == 0) {

                DPRINTF("NOP\n");

                break;

            }

            if (s->sist1 & LSI_SIST1_STO) {

                DPRINTF("Delayed select timeout\n");

                lsi_stop_script(s);

                break;

            }

            cond = jmp = (insn & (1 << 19)) != 0;

            if (cond == jmp && (insn & (1 << 21))) {

                DPRINTF("Compare carry %d\n", s->carry == jmp);

                cond = s->carry != 0;

            }

            if (cond == jmp && (insn & (1 << 17))) {

                DPRINTF("Compare phase %d %c= %d\n",

                        (s->sstat1 & PHASE_MASK),

                        jmp ? '=' : '!',

                        ((insn >> 24) & 7));

                cond = (s->sstat1 & PHASE_MASK) == ((insn >> 24) & 7);

            }

            if (cond == jmp && (insn & (1 << 18))) {

                uint8_t mask;



                mask = (~insn >> 8) & 0xff;

                DPRINTF("Compare data 0x%x & 0x%x %c= 0x%x\n",

                        s->sfbr, mask, jmp ? '=' : '!', insn & mask);

                cond = (s->sfbr & mask) == (insn & mask);

            }

            if (cond == jmp) {

                if (insn & (1 << 23)) {

                    /* Relative address.  */

                    addr = s->dsp + sxt24(addr);

                }

                switch ((insn >> 27) & 7) {

                case 0: /* Jump */

                    DPRINTF("Jump to 0x%08x\n", addr);

                    s->dsp = addr;

                    break;

                case 1: /* Call */

                    DPRINTF("Call 0x%08x\n", addr);

                    s->temp = s->dsp;

                    s->dsp = addr;

                    break;

                case 2: /* Return */

                    DPRINTF("Return to 0x%08x\n", s->temp);

                    s->dsp = s->temp;

                    break;

                case 3: /* Interrupt */

                    DPRINTF("Interrupt 0x%08x\n", s->dsps);

                    if ((insn & (1 << 20)) != 0) {

                        s->istat0 |= LSI_ISTAT0_INTF;

                        lsi_update_irq(s);

                    } else {

                        lsi_script_dma_interrupt(s, LSI_DSTAT_SIR);

                    }

                    break;

                default:

                    DPRINTF("Illegal transfer control\n");

                    lsi_script_dma_interrupt(s, LSI_DSTAT_IID);

                    break;

                }

            } else {

                DPRINTF("Control condition failed\n");

            }

        }

        break;



    case 3:

        if ((insn & (1 << 29)) == 0) {

            /* Memory move.  */

            uint32_t dest;

            /* ??? The docs imply the destination address is loaded into

               the TEMP register.  However the Linux drivers rely on

               the value being presrved.  */

            dest = read_dword(s, s->dsp);

            s->dsp += 4;

            lsi_memcpy(s, dest, addr, insn & 0xffffff);

        } else {

            uint8_t data[7];

            int reg;

            int n;

            int i;



            if (insn & (1 << 28)) {

                addr = s->dsa + sxt24(addr);

            }

            n = (insn & 7);

            reg = (insn >> 16) & 0xff;

            if (insn & (1 << 24)) {

                cpu_physical_memory_read(addr, data, n);

                DPRINTF("Load reg 0x%x size %d addr 0x%08x = %08x\n", reg, n,

                        addr, *(int *)data);

                for (i = 0; i < n; i++) {

                    lsi_reg_writeb(s, reg + i, data[i]);

                }

            } else {

                DPRINTF("Store reg 0x%x size %d addr 0x%08x\n", reg, n, addr);

                for (i = 0; i < n; i++) {

                    data[i] = lsi_reg_readb(s, reg + i);

                }

                cpu_physical_memory_write(addr, data, n);

            }

        }

    }

    /* ??? Need to avoid infinite loops.  */

    if (s->istat1 & LSI_ISTAT1_SRUN && !s->waiting) {

        if (s->dcntl & LSI_DCNTL_SSM) {

            lsi_script_dma_interrupt(s, LSI_DSTAT_SSI);

        } else {

            goto again;

        }

    }

    DPRINTF("SCRIPTS execution stopped\n");

}
