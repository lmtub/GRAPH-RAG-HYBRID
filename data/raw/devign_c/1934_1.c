static void usage(void)

{

    printf("qemu-" TARGET_ARCH " version " QEMU_VERSION QEMU_PKGVERSION ", Copyright (c) 2003-2008 Fabrice Bellard\n"

           "usage: qemu-" TARGET_ARCH " [options] program [arguments...]\n"

           "Linux CPU emulator (compiled for %s emulation)\n"

           "\n"

           "Standard options:\n"

           "-h                print this help\n"

           "-g port           wait gdb connection to port\n"

           "-L path           set the elf interpreter prefix (default=%s)\n"

           "-s size           set the stack size in bytes (default=%ld)\n"

           "-cpu model        select CPU (-cpu ? for list)\n"

           "-drop-ld-preload  drop LD_PRELOAD for target process\n"

           "-E var=value      sets/modifies targets environment variable(s)\n"

           "-U var            unsets targets environment variable(s)\n"

           "-0 argv0          forces target process argv[0] to be argv0\n"

#if defined(CONFIG_USE_GUEST_BASE)

           "-B address        set guest_base address to address\n"

#endif

           "\n"

           "Debug options:\n"

           "-d options   activate log (logfile=%s)\n"

           "-p pagesize  set the host page size to 'pagesize'\n"

           "-singlestep  always run in singlestep mode\n"

           "-strace      log system calls\n"

           "\n"

           "Environment variables:\n"

           "QEMU_STRACE       Print system calls and arguments similar to the\n"

           "                  'strace' program.  Enable by setting to any value.\n"

           "You can use -E and -U options to set/unset environment variables\n"

           "for target process.  It is possible to provide several variables\n"

           "by repeating the option.  For example:\n"

           "    -E var1=val2 -E var2=val2 -U LD_PRELOAD -U LD_DEBUG\n"

           "Note that if you provide several changes to single variable\n"

           "last change will stay in effect.\n"

           ,

           TARGET_ARCH,

           interp_prefix,

           x86_stack_size,

           DEBUG_LOGFILE);

    exit(1);

}
