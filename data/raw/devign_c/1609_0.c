unsigned int SingleCPDO(const unsigned int opcode)

{

   FPA11 *fpa11 = GET_FPA11();

   float32 rFm, rFn = 0;

   unsigned int Fd, Fm, Fn, nRc = 1;



   Fm = getFm(opcode);

   if (CONSTANT_FM(opcode))

   {

     rFm = getSingleConstant(Fm);

   }

   else

   {

     switch (fpa11->fType[Fm])

     {

        case typeSingle:

          rFm = fpa11->fpreg[Fm].fSingle;

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

          rFn = fpa11->fpreg[Fn].fSingle;

        break;



        default: return 0;

      }

   }



   Fd = getFd(opcode);

   switch (opcode & MASK_ARITHMETIC_OPCODE)

   {

      /* dyadic opcodes */

      case ADF_CODE:

         fpa11->fpreg[Fd].fSingle = float32_add(rFn,rFm, &fpa11->fp_status);

      break;



      case MUF_CODE:

      case FML_CODE:

        fpa11->fpreg[Fd].fSingle = float32_mul(rFn,rFm, &fpa11->fp_status);

      break;



      case SUF_CODE:

         fpa11->fpreg[Fd].fSingle = float32_sub(rFn,rFm, &fpa11->fp_status);

      break;



      case RSF_CODE:

         fpa11->fpreg[Fd].fSingle = float32_sub(rFm,rFn, &fpa11->fp_status);

      break;



      case DVF_CODE:

      case FDV_CODE:

         fpa11->fpreg[Fd].fSingle = float32_div(rFn,rFm, &fpa11->fp_status);

      break;



      case RDF_CODE:

      case FRD_CODE:

         fpa11->fpreg[Fd].fSingle = float32_div(rFm,rFn, &fpa11->fp_status);

      break;



#if 0

      case POW_CODE:

         fpa11->fpreg[Fd].fSingle = float32_pow(rFn,rFm);

      break;



      case RPW_CODE:

         fpa11->fpreg[Fd].fSingle = float32_pow(rFm,rFn);

      break;

#endif



      case RMF_CODE:

         fpa11->fpreg[Fd].fSingle = float32_rem(rFn,rFm, &fpa11->fp_status);

      break;



#if 0

      case POL_CODE:

         fpa11->fpreg[Fd].fSingle = float32_pol(rFn,rFm);

      break;

#endif



      /* monadic opcodes */

      case MVF_CODE:

         fpa11->fpreg[Fd].fSingle = rFm;

      break;



      case MNF_CODE:

         rFm ^= 0x80000000;

         fpa11->fpreg[Fd].fSingle = rFm;

      break;



      case ABS_CODE:

         rFm &= 0x7fffffff;

         fpa11->fpreg[Fd].fSingle = rFm;

      break;



      case RND_CODE:

      case URD_CODE:

         fpa11->fpreg[Fd].fSingle = float32_round_to_int(rFm, &fpa11->fp_status);

      break;



      case SQT_CODE:

         fpa11->fpreg[Fd].fSingle = float32_sqrt(rFm, &fpa11->fp_status);

      break;



#if 0

      case LOG_CODE:

         fpa11->fpreg[Fd].fSingle = float32_log(rFm);

      break;



      case LGN_CODE:

         fpa11->fpreg[Fd].fSingle = float32_ln(rFm);

      break;



      case EXP_CODE:

         fpa11->fpreg[Fd].fSingle = float32_exp(rFm);

      break;



      case SIN_CODE:

         fpa11->fpreg[Fd].fSingle = float32_sin(rFm);

      break;



      case COS_CODE:

         fpa11->fpreg[Fd].fSingle = float32_cos(rFm);

      break;



      case TAN_CODE:

         fpa11->fpreg[Fd].fSingle = float32_tan(rFm);

      break;



      case ASN_CODE:

         fpa11->fpreg[Fd].fSingle = float32_arcsin(rFm);

      break;



      case ACS_CODE:

         fpa11->fpreg[Fd].fSingle = float32_arccos(rFm);

      break;



      case ATN_CODE:

         fpa11->fpreg[Fd].fSingle = float32_arctan(rFm);

      break;

#endif



      case NRM_CODE:

      break;



      default:

      {

        nRc = 0;

      }

   }



   if (0 != nRc) fpa11->fType[Fd] = typeSingle;

   return nRc;

}
