print_insn (bfd_vma pc, disassemble_info *info)

{

  const struct dis386 *dp;

  int i;

  char *op_txt[MAX_OPERANDS];

  int needcomma;

  unsigned char uses_DATA_prefix, uses_LOCK_prefix;

  unsigned char uses_REPNZ_prefix, uses_REPZ_prefix;

  int sizeflag;

  const char *p;

  struct dis_private priv;

  unsigned char op;

  unsigned char threebyte;



  if (info->mach == bfd_mach_x86_64_intel_syntax

      || info->mach == bfd_mach_x86_64)

    address_mode = mode_64bit;

  else

    address_mode = mode_32bit;



  if (intel_syntax == (char) -1)

    intel_syntax = (info->mach == bfd_mach_i386_i386_intel_syntax

		    || info->mach == bfd_mach_x86_64_intel_syntax);



  if (info->mach == bfd_mach_i386_i386

      || info->mach == bfd_mach_x86_64

      || info->mach == bfd_mach_i386_i386_intel_syntax

      || info->mach == bfd_mach_x86_64_intel_syntax)

    priv.orig_sizeflag = AFLAG | DFLAG;

  else if (info->mach == bfd_mach_i386_i8086)

    priv.orig_sizeflag = 0;

  else

    abort ();



  for (p = info->disassembler_options; p != NULL; )

    {

      if (strncmp (p, "x86-64", 6) == 0)

	{

	  address_mode = mode_64bit;

	  priv.orig_sizeflag = AFLAG | DFLAG;

	}

      else if (strncmp (p, "i386", 4) == 0)

	{

	  address_mode = mode_32bit;

	  priv.orig_sizeflag = AFLAG | DFLAG;

	}

      else if (strncmp (p, "i8086", 5) == 0)

	{

	  address_mode = mode_16bit;

	  priv.orig_sizeflag = 0;

	}

      else if (strncmp (p, "intel", 5) == 0)

	{

	  intel_syntax = 1;

	}

      else if (strncmp (p, "att", 3) == 0)

	{

	  intel_syntax = 0;

	}

      else if (strncmp (p, "addr", 4) == 0)

	{

	  if (address_mode == mode_64bit)

	    {

	      if (p[4] == '3' && p[5] == '2')

		priv.orig_sizeflag &= ~AFLAG;

	      else if (p[4] == '6' && p[5] == '4')

		priv.orig_sizeflag |= AFLAG;

	    }

	  else

	    {

	      if (p[4] == '1' && p[5] == '6')

		priv.orig_sizeflag &= ~AFLAG;

	      else if (p[4] == '3' && p[5] == '2')

		priv.orig_sizeflag |= AFLAG;

	    }

	}

      else if (strncmp (p, "data", 4) == 0)

	{

	  if (p[4] == '1' && p[5] == '6')

	    priv.orig_sizeflag &= ~DFLAG;

	  else if (p[4] == '3' && p[5] == '2')

	    priv.orig_sizeflag |= DFLAG;

	}

      else if (strncmp (p, "suffix", 6) == 0)

	priv.orig_sizeflag |= SUFFIX_ALWAYS;



      p = strchr (p, ',');

      if (p != NULL)

	p++;

    }



  if (intel_syntax)

    {

      names64 = intel_names64;

      names32 = intel_names32;

      names16 = intel_names16;

      names8 = intel_names8;

      names8rex = intel_names8rex;

      names_seg = intel_names_seg;

      index16 = intel_index16;

      open_char = '[';

      close_char = ']';

      separator_char = '+';

      scale_char = '*';

    }

  else

    {

      names64 = att_names64;

      names32 = att_names32;

      names16 = att_names16;

      names8 = att_names8;

      names8rex = att_names8rex;

      names_seg = att_names_seg;

      index16 = att_index16;

      open_char = '(';

      close_char =  ')';

      separator_char = ',';

      scale_char = ',';

    }



  /* The output looks better if we put 7 bytes on a line, since that

     puts most long word instructions on a single line.  */

  info->bytes_per_line = 7;



  info->private_data = &priv;

  priv.max_fetched = priv.the_buffer;

  priv.insn_start = pc;



  obuf[0] = 0;

  for (i = 0; i < MAX_OPERANDS; ++i)

    {

      op_out[i][0] = 0;

      op_index[i] = -1;

    }



  the_info = info;

  start_pc = pc;

  start_codep = priv.the_buffer;

  codep = priv.the_buffer;



  if (sigsetjmp(priv.bailout, 0) != 0)

    {

      const char *name;



      /* Getting here means we tried for data but didn't get it.  That

	 means we have an incomplete instruction of some sort.  Just

	 print the first byte as a prefix or a .byte pseudo-op.  */

      if (codep > priv.the_buffer)

	{

	  name = prefix_name (priv.the_buffer[0], priv.orig_sizeflag);

	  if (name != NULL)

	    (*info->fprintf_func) (info->stream, "%s", name);

	  else

	    {

	      /* Just print the first byte as a .byte instruction.  */

	      (*info->fprintf_func) (info->stream, ".byte 0x%x",

				     (unsigned int) priv.the_buffer[0]);

	    }



	  return 1;

	}



      return -1;

    }



  obufp = obuf;

  ckprefix ();

  ckvexprefix ();



  insn_codep = codep;

  sizeflag = priv.orig_sizeflag;



  fetch_data(info, codep + 1);

  two_source_ops = (*codep == 0x62) || (*codep == 0xc8);



  if (((prefixes & PREFIX_FWAIT)

       && ((*codep < 0xd8) || (*codep > 0xdf)))

      || (rex && rex_used))

    {

      const char *name;



      /* fwait not followed by floating point instruction, or rex followed

	 by other prefixes.  Print the first prefix.  */

      name = prefix_name (priv.the_buffer[0], priv.orig_sizeflag);

      if (name == NULL)

	name = INTERNAL_DISASSEMBLER_ERROR;

      (*info->fprintf_func) (info->stream, "%s", name);

      return 1;

    }



  op = 0;

  if (prefixes & PREFIX_VEX_0F)

    {

      used_prefixes |= PREFIX_VEX_0F | PREFIX_VEX_0F38 | PREFIX_VEX_0F3A;

      if (prefixes & PREFIX_VEX_0F38)

        threebyte = 0x38;

      else if (prefixes & PREFIX_VEX_0F3A)

        threebyte = 0x3a;

      else

        threebyte = *codep++;

      goto vex_opcode;

    }

  if (*codep == 0x0f)

    {

      fetch_data(info, codep + 2);

      threebyte = codep[1];

      codep += 2;

    vex_opcode:

      dp = &dis386_twobyte[threebyte];

      need_modrm = twobyte_has_modrm[threebyte];

      uses_DATA_prefix = twobyte_uses_DATA_prefix[threebyte];

      uses_REPNZ_prefix = twobyte_uses_REPNZ_prefix[threebyte];

      uses_REPZ_prefix = twobyte_uses_REPZ_prefix[threebyte];

      uses_LOCK_prefix = (threebyte & ~0x02) == 0x20;

      if (dp->name == NULL && dp->op[0].bytemode == IS_3BYTE_OPCODE)

	{

          fetch_data(info, codep + 2);

	  op = *codep++;

	  switch (threebyte)

	    {

	    case 0x38:

	      uses_DATA_prefix = threebyte_0x38_uses_DATA_prefix[op];

	      uses_REPNZ_prefix = threebyte_0x38_uses_REPNZ_prefix[op];

	      uses_REPZ_prefix = threebyte_0x38_uses_REPZ_prefix[op];

	      break;

	    case 0x3a:

	      uses_DATA_prefix = threebyte_0x3a_uses_DATA_prefix[op];

	      uses_REPNZ_prefix = threebyte_0x3a_uses_REPNZ_prefix[op];

	      uses_REPZ_prefix = threebyte_0x3a_uses_REPZ_prefix[op];

	      break;

	    default:

	      break;

	    }

	}

    }

  else

    {

      dp = &dis386[*codep];

      need_modrm = onebyte_has_modrm[*codep];

      uses_DATA_prefix = 0;

      uses_REPNZ_prefix = 0;

      /* pause is 0xf3 0x90.  */

      uses_REPZ_prefix = *codep == 0x90;

      uses_LOCK_prefix = 0;

      codep++;

    }



  if (!uses_REPZ_prefix && (prefixes & PREFIX_REPZ))

    {

      oappend ("repz ");

      used_prefixes |= PREFIX_REPZ;

    }

  if (!uses_REPNZ_prefix && (prefixes & PREFIX_REPNZ))

    {

      oappend ("repnz ");

      used_prefixes |= PREFIX_REPNZ;

    }



  if (!uses_LOCK_prefix && (prefixes & PREFIX_LOCK))

    {

      oappend ("lock ");

      used_prefixes |= PREFIX_LOCK;

    }



  if (prefixes & PREFIX_ADDR)

    {

      sizeflag ^= AFLAG;

      if (dp->op[2].bytemode != loop_jcxz_mode || intel_syntax)

	{

	  if ((sizeflag & AFLAG) || address_mode == mode_64bit)

	    oappend ("addr32 ");

	  else

	    oappend ("addr16 ");

	  used_prefixes |= PREFIX_ADDR;

	}

    }



  if (!uses_DATA_prefix && (prefixes & PREFIX_DATA))

    {

      sizeflag ^= DFLAG;

      if (dp->op[2].bytemode == cond_jump_mode

	  && dp->op[0].bytemode == v_mode

	  && !intel_syntax)

	{

	  if (sizeflag & DFLAG)

	    oappend ("data32 ");

	  else

	    oappend ("data16 ");

	  used_prefixes |= PREFIX_DATA;

	}

    }



  if (dp->name == NULL && dp->op[0].bytemode == IS_3BYTE_OPCODE)

    {

      dp = &three_byte_table[dp->op[1].bytemode][op];

      modrm.mod = (*codep >> 6) & 3;

      modrm.reg = (*codep >> 3) & 7;

      modrm.rm = *codep & 7;

    }

  else if (need_modrm)

    {

      fetch_data(info, codep + 1);

      modrm.mod = (*codep >> 6) & 3;

      modrm.reg = (*codep >> 3) & 7;

      modrm.rm = *codep & 7;

    }



  if (dp->name == NULL && dp->op[0].bytemode == FLOATCODE)

    {

      dofloat (sizeflag);

    }

  else

    {

      int index;

      if (dp->name == NULL)

	{

	  switch (dp->op[0].bytemode)

	    {

	    case USE_GROUPS:

	      dp = &grps[dp->op[1].bytemode][modrm.reg];

	      break;



	    case USE_PREFIX_USER_TABLE:

	      index = 0;

	      used_prefixes |= (prefixes & PREFIX_REPZ);

	      if (prefixes & PREFIX_REPZ)

		index = 1;

	      else

		{

		  /* We should check PREFIX_REPNZ and PREFIX_REPZ

		     before PREFIX_DATA.  */

		  used_prefixes |= (prefixes & PREFIX_REPNZ);

		  if (prefixes & PREFIX_REPNZ)

		    index = 3;

		  else

		    {

		      used_prefixes |= (prefixes & PREFIX_DATA);

		      if (prefixes & PREFIX_DATA)

			index = 2;

		    }

		}

	      dp = &prefix_user_table[dp->op[1].bytemode][index];

	      break;



	    case X86_64_SPECIAL:

	      index = address_mode == mode_64bit ? 1 : 0;

	      dp = &x86_64_table[dp->op[1].bytemode][index];

	      break;



	    default:

	      oappend (INTERNAL_DISASSEMBLER_ERROR);

	      break;

	    }

	}



      if (putop (dp->name, sizeflag) == 0)

        {

	  for (i = 0; i < MAX_OPERANDS; ++i)

	    {

	      obufp = op_out[i];

	      op_ad = MAX_OPERANDS - 1 - i;

	      if (dp->op[i].rtn)

		(*dp->op[i].rtn) (dp->op[i].bytemode, sizeflag);

	    }

	}

    }



  /* See if any prefixes were not used.  If so, print the first one

     separately.  If we don't do this, we'll wind up printing an

     instruction stream which does not precisely correspond to the

     bytes we are disassembling.  */

  if ((prefixes & ~used_prefixes) != 0)

    {

      const char *name;



      name = prefix_name (priv.the_buffer[0], priv.orig_sizeflag);

      if (name == NULL)

	name = INTERNAL_DISASSEMBLER_ERROR;

      (*info->fprintf_func) (info->stream, "%s", name);

      return 1;

    }

  if (rex & ~rex_used)

    {

      const char *name;

      name = prefix_name (rex | 0x40, priv.orig_sizeflag);

      if (name == NULL)

	name = INTERNAL_DISASSEMBLER_ERROR;

      (*info->fprintf_func) (info->stream, "%s ", name);

    }



  obufp = obuf + strlen (obuf);

  for (i = strlen (obuf); i < 6; i++)

    oappend (" ");

  oappend (" ");

  (*info->fprintf_func) (info->stream, "%s", obuf);



  /* The enter and bound instructions are printed with operands in the same

     order as the intel book; everything else is printed in reverse order.  */

  if (intel_syntax || two_source_ops)

    {

      bfd_vma riprel;



      for (i = 0; i < MAX_OPERANDS; ++i)

        op_txt[i] = op_out[i];



      for (i = 0; i < (MAX_OPERANDS >> 1); ++i)

	{

          op_ad = op_index[i];

          op_index[i] = op_index[MAX_OPERANDS - 1 - i];

          op_index[MAX_OPERANDS - 1 - i] = op_ad;

	  riprel = op_riprel[i];

	  op_riprel[i] = op_riprel [MAX_OPERANDS - 1 - i];

	  op_riprel[MAX_OPERANDS - 1 - i] = riprel;

	}

    }

  else

    {

      for (i = 0; i < MAX_OPERANDS; ++i)

        op_txt[MAX_OPERANDS - 1 - i] = op_out[i];

    }



  needcomma = 0;

  for (i = 0; i < MAX_OPERANDS; ++i)

    if (*op_txt[i])

      {

	if (needcomma)

	  (*info->fprintf_func) (info->stream, ",");

	if (op_index[i] != -1 && !op_riprel[i])

	  (*info->print_address_func) ((bfd_vma) op_address[op_index[i]], info);

	else

	  (*info->fprintf_func) (info->stream, "%s", op_txt[i]);

	needcomma = 1;

      }



  for (i = 0; i < MAX_OPERANDS; i++)

    if (op_index[i] != -1 && op_riprel[i])

      {

	(*info->fprintf_func) (info->stream, "        # ");

	(*info->print_address_func) ((bfd_vma) (start_pc + codep - start_codep

						+ op_address[op_index[i]]), info);

	break;

      }

  return codep - priv.the_buffer;

}
