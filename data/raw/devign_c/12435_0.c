unsigned int DoubleCPDO(const unsigned int opcode)

{

   FPA11 *fpa11 = GET_FPA11();

   float64 rFm, rFn = 0;

   unsigned int Fd, Fm, Fn, nRc = 1;



   //printk("DoubleCPDO(0x%08x)\n",opcode);



   Fm = getFm(opcode);

   if (CONSTANT_FM(opcode))

   {

     rFm = getDoubleConstant(Fm);

   }

   else

   {

     switch (fpa11->fType[Fm])

     {

        case typeSingle:

          rFm = float32_to_float64(fpa11->fpreg[Fm].fSingle, &fpa11->fp_status);

        break;



        case typeDouble:

          rFm = fpa11->fpreg[Fm].fDouble;

          break;



        case typeExtended:

            // !! patb

	    //printk("not implemented! why not?\n");

            //!! ScottB

            // should never get here, if extended involved

            // then other operand should be promoted then

            // ExtendedCPDO called.

            break;



        default: return 0;

     }

   }



   if (!MONADIC_INSTRUCTION(opcode))

   {

      Fn = getFn(opcode);

      switch (fpa11->fType[Fn])

      {

        case typeSingle:

          rFn = float32_to_float64(fpa11->fpreg[Fn].fSingle, &fpa11->fp_status);

        break;



        case typeDouble:

          rFn = fpa11->fpreg[Fn].fDouble;

        break;



        default: return 0;

      }

   }



   Fd = getFd(opcode);

   /* !! this switch isn't optimized; better (opcode & MASK_ARITHMETIC_OPCODE)>>24, sort of */

   switch (opcode & MASK_ARITHMETIC_OPCODE)

   {

      /* dyadic opcodes */

      case ADF_CODE:

         fpa11->fpreg[Fd].fDouble = float64_add(rFn,rFm, &fpa11->fp_status);

      break;



      case MUF_CODE:

      case FML_CODE:

         fpa11->fpreg[Fd].fDouble = float64_mul(rFn,rFm, &fpa11->fp_status);

      break;



      case SUF_CODE:

         fpa11->fpreg[Fd].fDouble = float64_sub(rFn,rFm, &fpa11->fp_status);

      break;



      case RSF_CODE:

         fpa11->fpreg[Fd].fDouble = float64_sub(rFm,rFn, &fpa11->fp_status);

      break;



      case DVF_CODE:

      case FDV_CODE:

         fpa11->fpreg[Fd].fDouble = float64_div(rFn,rFm, &fpa11->fp_status);

      break;



      case RDF_CODE:

      case FRD_CODE:

         fpa11->fpreg[Fd].fDouble = float64_div(rFm,rFn, &fpa11->fp_status);

      break;



#if 0

      case POW_CODE:

         fpa11->fpreg[Fd].fDouble = float64_pow(rFn,rFm);

      break;



      case RPW_CODE:

         fpa11->fpreg[Fd].fDouble = float64_pow(rFm,rFn);

      break;

#endif



      case RMF_CODE:

         fpa11->fpreg[Fd].fDouble = float64_rem(rFn,rFm, &fpa11->fp_status);

      break;



#if 0

      case POL_CODE:

         fpa11->fpreg[Fd].fDouble = float64_pol(rFn,rFm);

      break;

#endif



      /* monadic opcodes */

      case MVF_CODE:

         fpa11->fpreg[Fd].fDouble = rFm;

      break;



      case MNF_CODE:

      {

         unsigned int *p = (unsigned int*)&rFm;

#ifdef WORDS_BIGENDIAN

         p[0] ^= 0x80000000;

#else

         p[1] ^= 0x80000000;

#endif

         fpa11->fpreg[Fd].fDouble = rFm;

      }

      break;



      case ABS_CODE:

      {

         unsigned int *p = (unsigned int*)&rFm;

#ifdef WORDS_BIGENDIAN

         p[0] &= 0x7fffffff;

#else

         p[1] &= 0x7fffffff;

#endif

         fpa11->fpreg[Fd].fDouble = rFm;

      }

      break;



      case RND_CODE:

      case URD_CODE:

         fpa11->fpreg[Fd].fDouble = float64_round_to_int(rFm, &fpa11->fp_status);

      break;



      case SQT_CODE:

         fpa11->fpreg[Fd].fDouble = float64_sqrt(rFm, &fpa11->fp_status);

      break;



#if 0

      case LOG_CODE:

         fpa11->fpreg[Fd].fDouble = float64_log(rFm);

      break;



      case LGN_CODE:

         fpa11->fpreg[Fd].fDouble = float64_ln(rFm);

      break;



      case EXP_CODE:

         fpa11->fpreg[Fd].fDouble = float64_exp(rFm);

      break;



      case SIN_CODE:

         fpa11->fpreg[Fd].fDouble = float64_sin(rFm);

      break;



      case COS_CODE:

         fpa11->fpreg[Fd].fDouble = float64_cos(rFm);

      break;



      case TAN_CODE:

         fpa11->fpreg[Fd].fDouble = float64_tan(rFm);

      break;



      case ASN_CODE:

         fpa11->fpreg[Fd].fDouble = float64_arcsin(rFm);

      break;



      case ACS_CODE:

         fpa11->fpreg[Fd].fDouble = float64_arccos(rFm);

      break;



      case ATN_CODE:

         fpa11->fpreg[Fd].fDouble = float64_arctan(rFm);

      break;

#endif



      case NRM_CODE:

      break;



      default:

      {

        nRc = 0;

      }

   }



   if (0 != nRc) fpa11->fType[Fd] = typeDouble;

   return nRc;

}
