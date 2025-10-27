int opt_cpuflags(const char *opt, const char *arg)

{

#define CPUFLAG_MMX2     (AV_CPU_FLAG_MMX      | AV_CPU_FLAG_MMX2)

#define CPUFLAG_3DNOW    (AV_CPU_FLAG_3DNOW    | AV_CPU_FLAG_MMX)

#define CPUFLAG_3DNOWEXT (AV_CPU_FLAG_3DNOWEXT | CPUFLAG_3DNOW)

#define CPUFLAG_SSE      (AV_CPU_FLAG_SSE      | CPUFLAG_MMX2)

#define CPUFLAG_SSE2     (AV_CPU_FLAG_SSE2     | CPUFLAG_SSE)

#define CPUFLAG_SSE2SLOW (AV_CPU_FLAG_SSE2SLOW | CPUFLAG_SSE2)

#define CPUFLAG_SSE3     (AV_CPU_FLAG_SSE3     | CPUFLAG_SSE2)

#define CPUFLAG_SSE3SLOW (AV_CPU_FLAG_SSE3SLOW | CPUFLAG_SSE3)

#define CPUFLAG_SSSE3    (AV_CPU_FLAG_SSSE3    | CPUFLAG_SSE3)

#define CPUFLAG_SSE4     (AV_CPU_FLAG_SSE4     | CPUFLAG_SSSE3)

#define CPUFLAG_SSE42    (AV_CPU_FLAG_SSE42    | CPUFLAG_SSE4)

#define CPUFLAG_AVX      (AV_CPU_FLAG_AVX      | CPUFLAG_SSE42)

#define CPUFLAG_XOP      (AV_CPU_FLAG_XOP      | CPUFLAG_AVX)

#define CPUFLAG_FMA4     (AV_CPU_FLAG_FMA4     | CPUFLAG_AVX)

    static const AVOption cpuflags_opts[] = {

        { "flags"   , NULL, 0, AV_OPT_TYPE_FLAGS, { 0 }, INT64_MIN, INT64_MAX, .unit = "flags" },

        { "altivec" , NULL, 0, AV_OPT_TYPE_CONST, { AV_CPU_FLAG_ALTIVEC  },    .unit = "flags" },

        { "mmx"     , NULL, 0, AV_OPT_TYPE_CONST, { AV_CPU_FLAG_MMX      },    .unit = "flags" },

        { "mmx2"    , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_MMX2         },    .unit = "flags" },

        { "sse"     , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_SSE          },    .unit = "flags" },

        { "sse2"    , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_SSE2         },    .unit = "flags" },

        { "sse2slow", NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_SSE2SLOW     },    .unit = "flags" },

        { "sse3"    , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_SSE3         },    .unit = "flags" },

        { "sse3slow", NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_SSE3SLOW     },    .unit = "flags" },

        { "ssse3"   , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_SSSE3        },    .unit = "flags" },

        { "atom"    , NULL, 0, AV_OPT_TYPE_CONST, { AV_CPU_FLAG_ATOM     },    .unit = "flags" },

        { "sse4.1"  , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_SSE4         },    .unit = "flags" },

        { "sse4.2"  , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_SSE42        },    .unit = "flags" },

        { "avx"     , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_AVX          },    .unit = "flags" },

        { "xop"     , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_XOP          },    .unit = "flags" },

        { "fma4"    , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_FMA4         },    .unit = "flags" },

        { "3dnow"   , NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_3DNOW        },    .unit = "flags" },

        { "3dnowext", NULL, 0, AV_OPT_TYPE_CONST, { CPUFLAG_3DNOWEXT     },    .unit = "flags" },

        { NULL },

    };

    static const AVClass class = {

        .class_name = "cpuflags",

        .item_name  = av_default_item_name,

        .option     = cpuflags_opts,

        .version    = LIBAVUTIL_VERSION_INT,

    };

    int flags = av_get_cpu_flags();

    int ret;

    const AVClass *pclass = &class;



    if ((ret = av_opt_eval_flags(&pclass, &cpuflags_opts[0], arg, &flags)) < 0)

        return ret;



    av_force_cpu_flags(flags);

    return 0;

}
