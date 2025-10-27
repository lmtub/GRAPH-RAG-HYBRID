int print_insn_lm32(bfd_vma memaddr, struct disassemble_info *info)

{

    fprintf_function fprintf_fn = info->fprintf_func;

    void *stream = info->stream;

    int rc;

    uint8_t insn[4];

    const Lm32OpcodeInfo *opc_info;

    uint32_t op;

    const char *args_fmt;



    rc = info->read_memory_func(memaddr, insn, 4, info);

    if (rc != 0) {

        info->memory_error_func(rc, memaddr, info);

        return -1;

    }



    fprintf_fn(stream, "%02x %02x %02x %02x    ",

            insn[0], insn[1], insn[2], insn[3]);



    op = bfd_getb32(insn);

    opc_info = find_opcode_info(op);

    if (opc_info) {

        fprintf_fn(stream, "%-8s ", opc_info->name);

        args_fmt = opc_info->args_fmt;

        while (args_fmt && *args_fmt) {

            if (*args_fmt == '%') {

                switch (*(++args_fmt)) {

                case '0': {

                    uint8_t r0;

                    const char *r0_name;

                    r0 = (op >> 21) & 0x1f;

                    r0_name = find_reg_info(r0)->name;

                    fprintf_fn(stream, "%s", r0_name);

                    break;

                }

                case '1': {

                    uint8_t r1;

                    const char *r1_name;

                    r1 = (op >> 16) & 0x1f;

                    r1_name = find_reg_info(r1)->name;

                    fprintf_fn(stream, "%s", r1_name);

                    break;

                }

                case '2': {

                    uint8_t r2;

                    const char *r2_name;

                    r2 = (op >> 11) & 0x1f;

                    r2_name = find_reg_info(r2)->name;

                    fprintf_fn(stream, "%s", r2_name);

                    break;

                }

                case 'c': {

                    uint8_t csr;

                    const char *csr_name;

                    csr = (op >> 21) & 0x1f;

                    csr_name = find_csr_info(csr)->name;

                    if (csr_name) {

                        fprintf_fn(stream, "%s", csr_name);

                    } else {

                        fprintf_fn(stream, "0x%x", csr);

                    }

                    break;

                }

                case 'u': {

                    uint16_t u16;

                    u16 = op & 0xffff;

                    fprintf_fn(stream, "0x%x", u16);

                    break;

                }

                case 's': {

                    int16_t s16;

                    s16 = (int16_t)(op & 0xffff);

                    fprintf_fn(stream, "%d", s16);

                    break;

                }

                case 'r': {

                    uint32_t rela;

                    rela = memaddr + (((int16_t)(op & 0xffff)) << 2);

                    fprintf_fn(stream, "%x", rela);

                    break;

                }

                case 'R': {

                    uint32_t rela;

                    int32_t imm26;

                    imm26 = (int32_t)((op & 0x3ffffff) << 6) >> 4;

                    rela = memaddr + imm26;

                    fprintf_fn(stream, "%x", rela);

                    break;

                }

                case 'h': {

                    uint8_t u5;

                    u5 = (op & 0x1f);

                    fprintf_fn(stream, "%d", u5);

                    break;

                }

                default:

                    break;

                }

            } else {

                fprintf_fn(stream, "%c", *args_fmt);

            }

            args_fmt++;

        }

    } else {

        fprintf_fn(stream, ".word 0x%x", op);

    }



    return 4;

}
