static void gen_check_sr(DisasContext *dc, uint32_t sr, unsigned access)

{

    if (!xtensa_option_bits_enabled(dc->config, sregnames[sr].opt_bits)) {

        if (sregnames[sr].name) {

            qemu_log("SR %s is not configured\n", sregnames[sr].name);

        } else {

            qemu_log("SR %d is not implemented\n", sr);

        }

        gen_exception_cause(dc, ILLEGAL_INSTRUCTION_CAUSE);

    } else if (!(sregnames[sr].access & access)) {

        static const char * const access_text[] = {

            [SR_R] = "rsr",

            [SR_W] = "wsr",

            [SR_X] = "xsr",

        };

        assert(access < ARRAY_SIZE(access_text) && access_text[access]);

        qemu_log("SR %s is not available for %s\n", sregnames[sr].name,

                access_text[access]);

        gen_exception_cause(dc, ILLEGAL_INSTRUCTION_CAUSE);

    }

}
