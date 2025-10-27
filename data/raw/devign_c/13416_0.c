match_insn_m68k (bfd_vma memaddr,

		 disassemble_info * info,

		 const struct m68k_opcode * best,

		 struct private * priv)

{

  unsigned char *save_p;

  unsigned char *p;

  const char *d;



  bfd_byte *buffer = priv->the_buffer;

  fprintf_ftype save_printer = info->fprintf_func;

  void (* save_print_address) (bfd_vma, struct disassemble_info *)

    = info->print_address_func;



  /* Point at first word of argument data,

     and at descriptor for first argument.  */

  p = buffer + 2;



  /* Figure out how long the fixed-size portion of the instruction is.

     The only place this is stored in the opcode table is

     in the arguments--look for arguments which specify fields in the 2nd

     or 3rd words of the instruction.  */

  for (d = best->args; *d; d += 2)

    {

      /* I don't think it is necessary to be checking d[0] here;

	 I suspect all this could be moved to the case statement below.  */

      if (d[0] == '#')

	{

	  if (d[1] == 'l' && p - buffer < 6)

	    p = buffer + 6;

	  else if (p - buffer < 4 && d[1] != 'C' && d[1] != '8')

	    p = buffer + 4;

	}



      if ((d[0] == 'L' || d[0] == 'l') && d[1] == 'w' && p - buffer < 4)

	p = buffer + 4;



      switch (d[1])

	{

	case '1':

	case '2':

	case '3':

	case '7':

	case '8':

	case '9':

	case 'i':

	  if (p - buffer < 4)

	    p = buffer + 4;

	  break;

	case '4':

	case '5':

	case '6':

	  if (p - buffer < 6)

	    p = buffer + 6;

	  break;

	default:

	  break;

	}

    }



  /* pflusha is an exceptions.  It takes no arguments but is two words

     long.  Recognize it by looking at the lower 16 bits of the mask.  */

  if (p - buffer < 4 && (best->match & 0xFFFF) != 0)

    p = buffer + 4;



  /* lpstop is another exception.  It takes a one word argument but is

     three words long.  */

  if (p - buffer < 6

      && (best->match & 0xffff) == 0xffff

      && best->args[0] == '#'

      && best->args[1] == 'w')

    {

      /* Copy the one word argument into the usual location for a one

	 word argument, to simplify printing it.  We can get away with

	 this because we know exactly what the second word is, and we

	 aren't going to print anything based on it.  */

      p = buffer + 6;

      FETCH_DATA (info, p);

      buffer[2] = buffer[4];

      buffer[3] = buffer[5];

    }



  FETCH_DATA (info, p);



  d = best->args;



  save_p = p;

  info->print_address_func = dummy_print_address;

  info->fprintf_func = (fprintf_ftype) dummy_printer;



  /* We scan the operands twice.  The first time we don't print anything,

     but look for errors.  */

  for (; *d; d += 2)

    {

      int eaten = print_insn_arg (d, buffer, p, memaddr + (p - buffer), info);



      if (eaten >= 0)

	p += eaten;

      else if (eaten == -1)

	{

	  info->fprintf_func = save_printer;

	  info->print_address_func = save_print_address;

	  return 0;

	}

      else

	{

	  info->fprintf_func (info->stream,

			      /* xgettext:c-format */

			      _("<internal error in opcode table: %s %s>\n"),

			      best->name,  best->args);

	  info->fprintf_func = save_printer;

	  info->print_address_func = save_print_address;

	  return 2;

	}

    }



  p = save_p;

  info->fprintf_func = save_printer;

  info->print_address_func = save_print_address;



  d = best->args;



  info->fprintf_func (info->stream, "%s", best->name);



  if (*d)

    info->fprintf_func (info->stream, " ");



  while (*d)

    {

      p += print_insn_arg (d, buffer, p, memaddr + (p - buffer), info);

      d += 2;



      if (*d && *(d - 2) != 'I' && *d != 'k')

	info->fprintf_func (info->stream, ",");

    }



  return p - buffer;

}
