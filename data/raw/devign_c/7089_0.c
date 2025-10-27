print_insn_microblaze (bfd_vma memaddr, struct disassemble_info * info)

{

  fprintf_ftype       fprintf = info->fprintf_func;

  void *              stream = info->stream;

  unsigned long       inst, prev_inst;

  struct op_code_struct * op, *pop;

  int                 immval = 0;

  bfd_boolean         immfound = FALSE;

  static bfd_vma prev_insn_addr = -1; /*init the prev insn addr */

  static int     prev_insn_vma = -1;  /*init the prev insn vma */

  int            curr_insn_vma = info->buffer_vma;



  info->bytes_per_chunk = 4;



  inst = read_insn_microblaze (memaddr, info, &op);

  if (inst == 0) {

    return -1;

  }

  

  if (prev_insn_vma == curr_insn_vma) {

  if (memaddr-(info->bytes_per_chunk) == prev_insn_addr) {

    prev_inst = read_insn_microblaze (prev_insn_addr, info, &pop);

    if (prev_inst == 0)

      return -1;

    if (pop->instr == imm) {

      immval = (get_int_field_imm(prev_inst) << 16) & 0xffff0000;

      immfound = TRUE;

    }

    else {

      immval = 0;

      immfound = FALSE;

    }

  }

  }

  /* make curr insn as prev insn */

  prev_insn_addr = memaddr;

  prev_insn_vma = curr_insn_vma;



  if (op->name == 0) {

    fprintf (stream, ".short 0x%04lx", inst);

  }

  else

    {

      fprintf (stream, "%s", op->name);

      

      switch (op->inst_type)

	{

  case INST_TYPE_RD_R1_R2:

     fprintf(stream, "\t%s, %s, %s", get_field_rd(inst), get_field_r1(inst), get_field_r2(inst));

     break;

        case INST_TYPE_RD_R1_IMM:

	  fprintf(stream, "\t%s, %s, %s", get_field_rd(inst), get_field_r1(inst), get_field_imm(inst));

	  if (info->print_address_func && get_int_field_r1(inst) == 0 && info->symbol_at_address_func) {

	    if (immfound)

	      immval |= (get_int_field_imm(inst) & 0x0000ffff);

	    else {

	      immval = get_int_field_imm(inst);

	      if (immval & 0x8000)

		immval |= 0xFFFF0000;

	    }

	    if (immval > 0 && info->symbol_at_address_func(immval, info)) {

	      fprintf (stream, "\t// ");

	      info->print_address_func (immval, info);

	    }

	  }

	  break;

	case INST_TYPE_RD_R1_IMM5:

	  fprintf(stream, "\t%s, %s, %s", get_field_rd(inst), get_field_r1(inst), get_field_imm5(inst));

	  break;

	case INST_TYPE_RD_RFSL:

	  fprintf(stream, "\t%s, %s", get_field_rd(inst), get_field_rfsl(inst));

	  break;

	case INST_TYPE_R1_RFSL:

	  fprintf(stream, "\t%s, %s", get_field_r1(inst), get_field_rfsl(inst));

	  break;

	case INST_TYPE_RD_SPECIAL:

	  fprintf(stream, "\t%s, %s", get_field_rd(inst), get_field_special(inst, op));

	  break;

	case INST_TYPE_SPECIAL_R1:

	  fprintf(stream, "\t%s, %s", get_field_special(inst, op), get_field_r1(inst));

	  break;

	case INST_TYPE_RD_R1:

	  fprintf(stream, "\t%s, %s", get_field_rd(inst), get_field_r1(inst));

	  break;

	case INST_TYPE_R1_R2:

	  fprintf(stream, "\t%s, %s", get_field_r1(inst), get_field_r2(inst));

	  break;

	case INST_TYPE_R1_IMM:

	  fprintf(stream, "\t%s, %s", get_field_r1(inst), get_field_imm(inst));

	  /* The non-pc relative instructions are returns, which shouldn't 

	     have a label printed */

	  if (info->print_address_func && op->inst_offset_type == INST_PC_OFFSET && info->symbol_at_address_func) {

	    if (immfound)

	      immval |= (get_int_field_imm(inst) & 0x0000ffff);

	    else {

	      immval = get_int_field_imm(inst);

	      if (immval & 0x8000)

		immval |= 0xFFFF0000;

	    }

	    immval += memaddr;

	    if (immval > 0 && info->symbol_at_address_func(immval, info)) {

	      fprintf (stream, "\t// ");

	      info->print_address_func (immval, info);

	    } else {

	      fprintf (stream, "\t\t// ");

	      fprintf (stream, "%x", immval);

	    }

	  }

	  break;

        case INST_TYPE_RD_IMM:

	  fprintf(stream, "\t%s, %s", get_field_rd(inst), get_field_imm(inst));

	  if (info->print_address_func && info->symbol_at_address_func) {

	    if (immfound)

	      immval |= (get_int_field_imm(inst) & 0x0000ffff);

	    else {

	      immval = get_int_field_imm(inst);

	      if (immval & 0x8000)

		immval |= 0xFFFF0000;

	    }

	    if (op->inst_offset_type == INST_PC_OFFSET)

	      immval += (int) memaddr;

	    if (info->symbol_at_address_func(immval, info)) {

	      fprintf (stream, "\t// ");

	      info->print_address_func (immval, info);

	    } 

	  }

	  break;

        case INST_TYPE_IMM:

	  fprintf(stream, "\t%s", get_field_imm(inst));

	  if (info->print_address_func && info->symbol_at_address_func && op->instr != imm) {

	    if (immfound)

	      immval |= (get_int_field_imm(inst) & 0x0000ffff);

	    else {

	      immval = get_int_field_imm(inst);

	      if (immval & 0x8000)

		immval |= 0xFFFF0000;

	    }

	    if (op->inst_offset_type == INST_PC_OFFSET)

	      immval += (int) memaddr;

	    if (immval > 0 && info->symbol_at_address_func(immval, info)) {

	      fprintf (stream, "\t// ");

	      info->print_address_func (immval, info);

	    } else if (op->inst_offset_type == INST_PC_OFFSET) {

	      fprintf (stream, "\t\t// ");

	      fprintf (stream, "%x", immval);

	    }

	  }

	  break;

        case INST_TYPE_RD_R2:

	  fprintf(stream, "\t%s, %s", get_field_rd(inst), get_field_r2(inst));

	  break;

  case INST_TYPE_R2:

     fprintf(stream, "\t%s", get_field_r2(inst));

     break;

  case INST_TYPE_R1:

     fprintf(stream, "\t%s", get_field_r1(inst));

     break;

  case INST_TYPE_RD_R1_SPECIAL:

     fprintf(stream, "\t%s, %s", get_field_rd(inst), get_field_r2(inst));

     break;

  case INST_TYPE_RD_IMM15:

     fprintf(stream, "\t%s, %s", get_field_rd(inst), get_field_imm15(inst));

     break;

     /* For tuqula instruction */

  case INST_TYPE_RD:

     fprintf(stream, "\t%s", get_field_rd(inst));

     break;

  case INST_TYPE_RFSL:

     fprintf(stream, "\t%s", get_field_rfsl(inst));

     break;

  default:

	  /* if the disassembler lags the instruction set */

	  fprintf (stream, "\tundecoded operands, inst is 0x%04lx", inst);

	  break;

	}

    }

  

  /* Say how many bytes we consumed? */

  return 4;

}
