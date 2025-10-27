void fpu_dump_state(CPUState *env, FILE *f, 

                    int (*fpu_fprintf)(FILE *f, const char *fmt, ...),

                    int flags)

{

    int i;

    int is_fpu64 = !!(env->CP0_Status & (1 << CP0St_FR));



#define printfpr(fp)                                                        \

    do {                                                                    \

        if (is_fpu64)                                                       \

            fpu_fprintf(f, "w:%08x d:%016lx fd:%13g fs:%13g psu: %13g\n",   \

                        (fp)->w[FP_ENDIAN_IDX], (fp)->d, (fp)->fd,          \

                        (fp)->fs[FP_ENDIAN_IDX], (fp)->fs[!FP_ENDIAN_IDX]); \

        else {                                                              \

            fpr_t tmp;                                                      \

            tmp.w[FP_ENDIAN_IDX] = (fp)->w[FP_ENDIAN_IDX];                  \

            tmp.w[!FP_ENDIAN_IDX] = ((fp) + 1)->w[FP_ENDIAN_IDX];           \

            fpu_fprintf(f, "w:%08x d:%016lx fd:%13g fs:%13g psu:%13g\n",    \

                        tmp.w[FP_ENDIAN_IDX], tmp.d, tmp.fd,                \

                        tmp.fs[FP_ENDIAN_IDX], tmp.fs[!FP_ENDIAN_IDX]);     \

        }                                                                   \

    } while(0)





    fpu_fprintf(f, "CP1 FCR0 0x%08x  FCR31 0x%08x  SR.FR %d  fp_status 0x%08x(0x%02x)\n",

                env->fcr0, env->fcr31, is_fpu64, env->fp_status, get_float_exception_flags(&env->fp_status));

    fpu_fprintf(f, "FT0: "); printfpr(&env->ft0);

    fpu_fprintf(f, "FT1: "); printfpr(&env->ft1);

    fpu_fprintf(f, "FT2: "); printfpr(&env->ft2);

    for (i = 0; i < 32; (is_fpu64) ? i++ : (i += 2)) {

        fpu_fprintf(f, "%3s: ", fregnames[i]);

        printfpr(&env->fpr[i]);

    }



#undef printfpr

}
