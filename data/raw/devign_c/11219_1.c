print_operand_value (char *buf, size_t bufsize, int hex, bfd_vma disp)

{

  if (address_mode == mode_64bit)

    {

      if (hex)

	{

	  char tmp[30];

	  int i;

	  buf[0] = '0';

	  buf[1] = 'x';

          snprintf_vma (tmp, sizeof(tmp), disp);

	  for (i = 0; tmp[i] == '0' && tmp[i + 1]; i++);

          pstrcpy (buf + 2, bufsize - 2, tmp + i);

	}

      else

	{

	  bfd_signed_vma v = disp;

	  char tmp[30];

	  int i;

	  if (v < 0)

	    {

	      *(buf++) = '-';

	      v = -disp;

	      /* Check for possible overflow on 0x8000000000000000.  */

	      if (v < 0)

		{

                  pstrcpy (buf, bufsize, "9223372036854775808");

		  return;

		}

	    }

	  if (!v)

	    {

                pstrcpy (buf, bufsize, "0");

	      return;

	    }



	  i = 0;

	  tmp[29] = 0;

	  while (v)

	    {

	      tmp[28 - i] = (v % 10) + '0';

	      v /= 10;

	      i++;

	    }

          pstrcpy (buf, bufsize, tmp + 29 - i);

	}

    }

  else

    {

      if (hex)

        snprintf (buf, bufsize, "0x%x", (unsigned int) disp);

      else

        snprintf (buf, bufsize, "%d", (int) disp);

    }

}
