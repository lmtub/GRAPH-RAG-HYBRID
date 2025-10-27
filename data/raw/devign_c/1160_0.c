static void usage(void)

{

    const struct qemu_argument *arginfo;

    int maxarglen;

    int maxenvlen;



    printf("usage: qemu-" TARGET_ARCH " [options] program [arguments...]\n"

           "Linux CPU emulator (compiled for " TARGET_ARCH " emulation)\n"

           "\n"

           "Options and associated environment variables:\n"

           "\n");



    maxarglen = maxenvlen = 0;



    for (arginfo = arg_table; arginfo->handle_opt != NULL; arginfo++) {

        if (strlen(arginfo->env) > maxenvlen) {

            maxenvlen = strlen(arginfo->env);

        }

        if (strlen(arginfo->argv) > maxarglen) {

            maxarglen = strlen(arginfo->argv);

        }

    }



    printf("%-*s%-*sDescription\n", maxarglen+3, "Argument",

            maxenvlen+1, "Env-variable");



    for (arginfo = arg_table; arginfo->handle_opt != NULL; arginfo++) {

        if (arginfo->has_arg) {

            printf("-%s %-*s %-*s %s\n", arginfo->argv,

                    (int)(maxarglen-strlen(arginfo->argv)), arginfo->example,

                    maxenvlen, arginfo->env, arginfo->help);

        } else {

            printf("-%-*s %-*s %s\n", maxarglen+1, arginfo->argv,

                    maxenvlen, arginfo->env,

                    arginfo->help);

        }

    }



    printf("\n"

           "Defaults:\n"

           "QEMU_LD_PREFIX  = %s\n"

           "QEMU_STACK_SIZE = %ld byte\n",

           interp_prefix,

           guest_stack_size);



    printf("\n"

           "You can use -E and -U options or the QEMU_SET_ENV and\n"

           "QEMU_UNSET_ENV environment variables to set and unset\n"

           "environment variables for the target process.\n"

           "It is possible to provide several variables by separating them\n"

           "by commas in getsubopt(3) style. Additionally it is possible to\n"

           "provide the -E and -U options multiple times.\n"

           "The following lines are equivalent:\n"

           "    -E var1=val2 -E var2=val2 -U LD_PRELOAD -U LD_DEBUG\n"

           "    -E var1=val2,var2=val2 -U LD_PRELOAD,LD_DEBUG\n"

           "    QEMU_SET_ENV=var1=val2,var2=val2 QEMU_UNSET_ENV=LD_PRELOAD,LD_DEBUG\n"

           "Note that if you provide several changes to a single variable\n"

           "the last change will stay in effect.\n");



    exit(1);

}
