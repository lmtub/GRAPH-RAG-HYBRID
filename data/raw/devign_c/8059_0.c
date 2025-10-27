print_insn_arg (const char *d,

		unsigned char *buffer,

		unsigned char *p0,

		bfd_vma addr,

		disassemble_info *info)

{

  int val = 0;

  int place = d[1];

  unsigned char *p = p0;

  int regno;

  const char *regname;

  unsigned char *p1;

  double flval;

  int flt_p;

  bfd_signed_vma disp;

  unsigned int uval;



  switch (*d)

    {

    case 'c':		/* Cache identifier.  */

      {

        static const char *const cacheFieldName[] = { "nc", "dc", "ic", "bc" };

        val = fetch_arg (buffer, place, 2, info);

        (*info->fprintf_func) (info->stream, cacheFieldName[val]);

        break;

      }



    case 'a':		/* Address register indirect only. Cf. case '+'.  */

      {

        (*info->fprintf_func)

	  (info->stream,

	   "%s@",

	   reg_names[fetch_arg (buffer, place, 3, info) + 8]);

        break;

      }



    case '_':		/* 32-bit absolute address for move16.  */

      {

        uval = NEXTULONG (p);

	(*info->print_address_func) (uval, info);

        break;

      }



    case 'C':

      (*info->fprintf_func) (info->stream, "%%ccr");

      break;



    case 'S':

      (*info->fprintf_func) (info->stream, "%%sr");

      break;



    case 'U':

      (*info->fprintf_func) (info->stream, "%%usp");

      break;



    case 'E':

      (*info->fprintf_func) (info->stream, "%%acc");

      break;



    case 'G':

      (*info->fprintf_func) (info->stream, "%%macsr");

      break;



    case 'H':

      (*info->fprintf_func) (info->stream, "%%mask");

      break;



    case 'J':

      {

	/* FIXME: There's a problem here, different m68k processors call the

	   same address different names. This table can't get it right

	   because it doesn't know which processor it's disassembling for.  */

	static const struct { const char *name; int value; } names[]

	  = {{"%sfc", 0x000}, {"%dfc", 0x001}, {"%cacr", 0x002},

	     {"%tc",  0x003}, {"%itt0",0x004}, {"%itt1", 0x005},

             {"%dtt0",0x006}, {"%dtt1",0x007}, {"%buscr",0x008},

	     {"%usp", 0x800}, {"%vbr", 0x801}, {"%caar", 0x802},

	     {"%msp", 0x803}, {"%isp", 0x804},

	     {"%flashbar", 0xc04}, {"%rambar", 0xc05}, /* mcf528x added these.  */



	     /* Should we be calling this psr like we do in case 'Y'?  */

	     {"%mmusr",0x805},



             {"%urp", 0x806}, {"%srp", 0x807}, {"%pcr", 0x808}};



	val = fetch_arg (buffer, place, 12, info);

	for (regno = sizeof names / sizeof names[0] - 1; regno >= 0; regno--)

	  if (names[regno].value == val)

	    {

	      (*info->fprintf_func) (info->stream, "%s", names[regno].name);

	      break;

	    }

	if (regno < 0)

	  (*info->fprintf_func) (info->stream, "%d", val);

      }

      break;



    case 'Q':

      val = fetch_arg (buffer, place, 3, info);

      /* 0 means 8, except for the bkpt instruction... */

      if (val == 0 && d[1] != 's')

	val = 8;

      (*info->fprintf_func) (info->stream, "#%d", val);

      break;



    case 'x':

      val = fetch_arg (buffer, place, 3, info);

      /* 0 means -1.  */

      if (val == 0)

	val = -1;

      (*info->fprintf_func) (info->stream, "#%d", val);

      break;



    case 'M':

      if (place == 'h')

	{

	  static const char *const scalefactor_name[] = { "<<", ">>" };

	  val = fetch_arg (buffer, place, 1, info);

	  (*info->fprintf_func) (info->stream, scalefactor_name[val]);

	}

      else

	{

	  val = fetch_arg (buffer, place, 8, info);

	  if (val & 0x80)

	    val = val - 0x100;

	  (*info->fprintf_func) (info->stream, "#%d", val);

	}

      break;



    case 'T':

      val = fetch_arg (buffer, place, 4, info);

      (*info->fprintf_func) (info->stream, "#%d", val);

      break;



    case 'D':

      (*info->fprintf_func) (info->stream, "%s",

			     reg_names[fetch_arg (buffer, place, 3, info)]);

      break;



    case 'A':

      (*info->fprintf_func)

	(info->stream, "%s",

	 reg_names[fetch_arg (buffer, place, 3, info) + 010]);

      break;



    case 'R':

      (*info->fprintf_func)

	(info->stream, "%s",

	 reg_names[fetch_arg (buffer, place, 4, info)]);

      break;



    case 'r':

      regno = fetch_arg (buffer, place, 4, info);

      if (regno > 7)

	(*info->fprintf_func) (info->stream, "%s@", reg_names[regno]);

      else

	(*info->fprintf_func) (info->stream, "@(%s)", reg_names[regno]);

      break;



    case 'F':

      (*info->fprintf_func)

	(info->stream, "%%fp%d",

	 fetch_arg (buffer, place, 3, info));

      break;



    case 'O':

      val = fetch_arg (buffer, place, 6, info);

      if (val & 0x20)

	(*info->fprintf_func) (info->stream, "%s", reg_names[val & 7]);

      else

	(*info->fprintf_func) (info->stream, "%d", val);

      break;



    case '+':

      (*info->fprintf_func)

	(info->stream, "%s@+",

	 reg_names[fetch_arg (buffer, place, 3, info) + 8]);

      break;



    case '-':

      (*info->fprintf_func)

	(info->stream, "%s@-",

	 reg_names[fetch_arg (buffer, place, 3, info) + 8]);

      break;



    case 'k':

      if (place == 'k')

	(*info->fprintf_func)

	  (info->stream, "{%s}",

	   reg_names[fetch_arg (buffer, place, 3, info)]);

      else if (place == 'C')

	{

	  val = fetch_arg (buffer, place, 7, info);

	  if (val > 63)		/* This is a signed constant.  */

	    val -= 128;

	  (*info->fprintf_func) (info->stream, "{#%d}", val);

	}

      else

	return -2;

      break;



    case '#':

    case '^':

      p1 = buffer + (*d == '#' ? 2 : 4);

      if (place == 's')

	val = fetch_arg (buffer, place, 4, info);

      else if (place == 'C')

	val = fetch_arg (buffer, place, 7, info);

      else if (place == '8')

	val = fetch_arg (buffer, place, 3, info);

      else if (place == '3')

	val = fetch_arg (buffer, place, 8, info);

      else if (place == 'b')

	val = NEXTBYTE (p1);

      else if (place == 'w' || place == 'W')

	val = NEXTWORD (p1);

      else if (place == 'l')

	val = NEXTLONG (p1);

      else

	return -2;

      (*info->fprintf_func) (info->stream, "#%d", val);

      break;



    case 'B':

      if (place == 'b')

	disp = NEXTBYTE (p);

      else if (place == 'B')

	disp = COERCE_SIGNED_CHAR (buffer[1]);

      else if (place == 'w' || place == 'W')

	disp = NEXTWORD (p);

      else if (place == 'l' || place == 'L' || place == 'C')

	disp = NEXTLONG (p);

      else if (place == 'g')

	{

	  disp = NEXTBYTE (buffer);

	  if (disp == 0)

	    disp = NEXTWORD (p);

	  else if (disp == -1)

	    disp = NEXTLONG (p);

	}

      else if (place == 'c')

	{

	  if (buffer[1] & 0x40)		/* If bit six is one, long offset.  */

	    disp = NEXTLONG (p);

	  else

	    disp = NEXTWORD (p);

	}

      else

	return -2;



      (*info->print_address_func) (addr + disp, info);

      break;



    case 'd':

      val = NEXTWORD (p);

      (*info->fprintf_func)

	(info->stream, "%s@(%d)",

	 reg_names[fetch_arg (buffer, place, 3, info) + 8], val);

      break;



    case 's':

      (*info->fprintf_func) (info->stream, "%s",

			     fpcr_names[fetch_arg (buffer, place, 3, info)]);

      break;



    case 'e':

      val = fetch_arg(buffer, place, 2, info);

      (*info->fprintf_func) (info->stream, "%%acc%d", val);

      break;



    case 'g':

      val = fetch_arg(buffer, place, 1, info);

      (*info->fprintf_func) (info->stream, "%%accext%s", val==0 ? "01" : "23");

      break;



    case 'i':

      val = fetch_arg(buffer, place, 2, info);

      if (val == 1)

	(*info->fprintf_func) (info->stream, "<<");

      else if (val == 3)

	(*info->fprintf_func) (info->stream, ">>");

      else

	return -1;

      break;



    case 'I':

      /* Get coprocessor ID... */

      val = fetch_arg (buffer, 'd', 3, info);



      if (val != 1)				/* Unusual coprocessor ID?  */

	(*info->fprintf_func) (info->stream, "(cpid=%d) ", val);

      break;



    case '4':

    case '*':

    case '~':

    case '%':

    case ';':

    case '@':

    case '!':

    case '$':

    case '?':

    case '/':

    case '&':

    case '|':

    case '<':

    case '>':

    case 'm':

    case 'n':

    case 'o':

    case 'p':

    case 'q':

    case 'v':

    case 'b':

    case 'w':

    case 'y':

    case 'z':

      if (place == 'd')

	{

	  val = fetch_arg (buffer, 'x', 6, info);

	  val = ((val & 7) << 3) + ((val >> 3) & 7);

	}

      else

	val = fetch_arg (buffer, 's', 6, info);



      /* If the <ea> is invalid for *d, then reject this match.  */

      if (!m68k_valid_ea (*d, val))

	return -1;



      /* Get register number assuming address register.  */

      regno = (val & 7) + 8;

      regname = reg_names[regno];

      switch (val >> 3)

	{

	case 0:

	  (*info->fprintf_func) (info->stream, "%s", reg_names[val]);

	  break;



	case 1:

	  (*info->fprintf_func) (info->stream, "%s", regname);

	  break;



	case 2:

	  (*info->fprintf_func) (info->stream, "%s@", regname);

	  break;



	case 3:

	  (*info->fprintf_func) (info->stream, "%s@+", regname);

	  break;



	case 4:

	  (*info->fprintf_func) (info->stream, "%s@-", regname);

	  break;



	case 5:

	  val = NEXTWORD (p);

	  (*info->fprintf_func) (info->stream, "%s@(%d)", regname, val);

	  break;



	case 6:

	  p = print_indexed (regno, p, addr, info);

	  break;



	case 7:

	  switch (val & 7)

	    {

	    case 0:

	      val = NEXTWORD (p);

	      (*info->print_address_func) (val, info);

	      break;



	    case 1:

	      uval = NEXTULONG (p);

	      (*info->print_address_func) (uval, info);

	      break;



	    case 2:

	      val = NEXTWORD (p);

	      (*info->fprintf_func) (info->stream, "%%pc@(");

	      (*info->print_address_func) (addr + val, info);

	      (*info->fprintf_func) (info->stream, ")");

	      break;



	    case 3:

	      p = print_indexed (-1, p, addr, info);

	      break;



	    case 4:

	      flt_p = 1;	/* Assume it's a float... */

	      switch (place)

	      {

		case 'b':

		  val = NEXTBYTE (p);

		  flt_p = 0;

		  break;



		case 'w':

		  val = NEXTWORD (p);

		  flt_p = 0;

		  break;



		case 'l':

		  val = NEXTLONG (p);

		  flt_p = 0;

		  break;



		case 'f':

		  NEXTSINGLE (flval, p);

		  break;



		case 'F':

		  NEXTDOUBLE (flval, p);

		  break;



		case 'x':

		  NEXTEXTEND (flval, p);

		  break;



		case 'p':

		  flval = NEXTPACKED (p);

		  break;



		default:

		  return -1;

	      }

	      if (flt_p)	/* Print a float? */

		(*info->fprintf_func) (info->stream, "#%g", flval);

	      else

		(*info->fprintf_func) (info->stream, "#%d", val);

	      break;



	    default:

	      return -1;

	    }

	}



      /* If place is '/', then this is the case of the mask bit for

	 mac/emac loads. Now that the arg has been printed, grab the

	 mask bit and if set, add a '&' to the arg.  */

      if (place == '/')

	{

	  val = fetch_arg (buffer, place, 1, info);

	  if (val)

	    info->fprintf_func (info->stream, "&");

	}

      break;



    case 'L':

    case 'l':

	if (place == 'w')

	  {

	    char doneany;

	    p1 = buffer + 2;

	    val = NEXTWORD (p1);

	    /* Move the pointer ahead if this point is farther ahead

	       than the last.  */

	    p = p1 > p ? p1 : p;

	    if (val == 0)

	      {

		(*info->fprintf_func) (info->stream, "#0");

		break;

	      }

	    if (*d == 'l')

	      {

		int newval = 0;



		for (regno = 0; regno < 16; ++regno)

		  if (val & (0x8000 >> regno))

		    newval |= 1 << regno;

		val = newval;

	      }

	    val &= 0xffff;

	    doneany = 0;

	    for (regno = 0; regno < 16; ++regno)

	      if (val & (1 << regno))

		{

		  int first_regno;



		  if (doneany)

		    (*info->fprintf_func) (info->stream, "/");

		  doneany = 1;

		  (*info->fprintf_func) (info->stream, "%s", reg_names[regno]);

		  first_regno = regno;

		  while (val & (1 << (regno + 1)))

		    ++regno;

		  if (regno > first_regno)

		    (*info->fprintf_func) (info->stream, "-%s",

					   reg_names[regno]);

		}

	  }

	else if (place == '3')

	  {

	    /* `fmovem' insn.  */

	    char doneany;

	    val = fetch_arg (buffer, place, 8, info);

	    if (val == 0)

	      {

		(*info->fprintf_func) (info->stream, "#0");

		break;

	      }

	    if (*d == 'l')

	      {

		int newval = 0;



		for (regno = 0; regno < 8; ++regno)

		  if (val & (0x80 >> regno))

		    newval |= 1 << regno;

		val = newval;

	      }

	    val &= 0xff;

	    doneany = 0;

	    for (regno = 0; regno < 8; ++regno)

	      if (val & (1 << regno))

		{

		  int first_regno;

		  if (doneany)

		    (*info->fprintf_func) (info->stream, "/");

		  doneany = 1;

		  (*info->fprintf_func) (info->stream, "%%fp%d", regno);

		  first_regno = regno;

		  while (val & (1 << (regno + 1)))

		    ++regno;

		  if (regno > first_regno)

		    (*info->fprintf_func) (info->stream, "-%%fp%d", regno);

		}

	  }

	else if (place == '8')

	  {

	    /* fmoveml for FP status registers.  */

	    (*info->fprintf_func) (info->stream, "%s",

				   fpcr_names[fetch_arg (buffer, place, 3,

							 info)]);

	  }

	else

	  return -2;

      break;



    case 'X':

      place = '8';

    case 'Y':

    case 'Z':

    case 'W':

    case '0':

    case '1':

    case '2':

    case '3':

      {

	int val = fetch_arg (buffer, place, 5, info);

        const char *name = 0;



	switch (val)

	  {

	  case 2: name = "%tt0"; break;

	  case 3: name = "%tt1"; break;

	  case 0x10: name = "%tc"; break;

	  case 0x11: name = "%drp"; break;

	  case 0x12: name = "%srp"; break;

	  case 0x13: name = "%crp"; break;

	  case 0x14: name = "%cal"; break;

	  case 0x15: name = "%val"; break;

	  case 0x16: name = "%scc"; break;

	  case 0x17: name = "%ac"; break;

 	  case 0x18: name = "%psr"; break;

	  case 0x19: name = "%pcsr"; break;

	  case 0x1c:

	  case 0x1d:

	    {

	      int break_reg = ((buffer[3] >> 2) & 7);



	      (*info->fprintf_func)

		(info->stream, val == 0x1c ? "%%bad%d" : "%%bac%d",

		 break_reg);

	    }

	    break;

	  default:

	    (*info->fprintf_func) (info->stream, "<mmu register %d>", val);

	  }

	if (name)

	  (*info->fprintf_func) (info->stream, "%s", name);

      }

      break;



    case 'f':

      {

	int fc = fetch_arg (buffer, place, 5, info);



	if (fc == 1)

	  (*info->fprintf_func) (info->stream, "%%dfc");

	else if (fc == 0)

	  (*info->fprintf_func) (info->stream, "%%sfc");

	else

	  /* xgettext:c-format */

	  (*info->fprintf_func) (info->stream, _("<function code %d>"), fc);

      }

      break;



    case 'V':

      (*info->fprintf_func) (info->stream, "%%val");

      break;



    case 't':

      {

	int level = fetch_arg (buffer, place, 3, info);



	(*info->fprintf_func) (info->stream, "%d", level);

      }

      break;



    case 'u':

      {

	short is_upper = 0;

	int reg = fetch_arg (buffer, place, 5, info);



	if (reg & 0x10)

	  {

	    is_upper = 1;

	    reg &= 0xf;

	  }

	(*info->fprintf_func) (info->stream, "%s%s",

			       reg_half_names[reg],

			       is_upper ? "u" : "l");

      }

      break;



    default:

      return -2;

    }



  return p - p0;

}
