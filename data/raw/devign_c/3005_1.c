static abi_ulong create_elf_tables(abi_ulong p, int argc, int envc,

                                   struct elfhdr *exec,

                                   struct image_info *info,

                                   struct image_info *interp_info)

{

    abi_ulong sp;

    abi_ulong u_argc, u_argv, u_envp, u_auxv;

    int size;

    int i;

    abi_ulong u_rand_bytes;

    uint8_t k_rand_bytes[16];

    abi_ulong u_platform;

    const char *k_platform;

    const int n = sizeof(elf_addr_t);



    sp = p;



#ifdef CONFIG_USE_FDPIC

    /* Needs to be before we load the env/argc/... */

    if (elf_is_fdpic(exec)) {

        /* Need 4 byte alignment for these structs */

        sp &= ~3;

        sp = loader_build_fdpic_loadmap(info, sp);

        info->other_info = interp_info;

        if (interp_info) {

            interp_info->other_info = info;

            sp = loader_build_fdpic_loadmap(interp_info, sp);

        }

    }

#endif



    u_platform = 0;

    k_platform = ELF_PLATFORM;

    if (k_platform) {

        size_t len = strlen(k_platform) + 1;

        if (STACK_GROWS_DOWN) {

            sp -= (len + n - 1) & ~(n - 1);

            u_platform = sp;

            /* FIXME - check return value of memcpy_to_target() for failure */

            memcpy_to_target(sp, k_platform, len);

        } else {

            memcpy_to_target(sp, k_platform, len);

            u_platform = sp;

            sp += len + 1;

        }

    }



    /* Provide 16 byte alignment for the PRNG, and basic alignment for

     * the argv and envp pointers.

     */

    if (STACK_GROWS_DOWN) {

        sp = QEMU_ALIGN_DOWN(sp, 16);

    } else {

        sp = QEMU_ALIGN_UP(sp, 16);

    }



    /*

     * Generate 16 random bytes for userspace PRNG seeding (not

     * cryptically secure but it's not the aim of QEMU).

     */

    for (i = 0; i < 16; i++) {

        k_rand_bytes[i] = rand();

    }

    if (STACK_GROWS_DOWN) {

        sp -= 16;

        u_rand_bytes = sp;

        /* FIXME - check return value of memcpy_to_target() for failure */

        memcpy_to_target(sp, k_rand_bytes, 16);

    } else {

        memcpy_to_target(sp, k_rand_bytes, 16);

        u_rand_bytes = sp;

        sp += 16;

    }



    size = (DLINFO_ITEMS + 1) * 2;

    if (k_platform)

        size += 2;

#ifdef DLINFO_ARCH_ITEMS

    size += DLINFO_ARCH_ITEMS * 2;

#endif

#ifdef ELF_HWCAP2

    size += 2;

#endif

    size += envc + argc + 2;

    size += 1;  /* argc itself */

    size *= n;



    /* Allocate space and finalize stack alignment for entry now.  */

    if (STACK_GROWS_DOWN) {

        u_argc = QEMU_ALIGN_DOWN(sp - size, STACK_ALIGNMENT);

        sp = u_argc;

    } else {

        u_argc = sp;

        sp = QEMU_ALIGN_UP(sp + size, STACK_ALIGNMENT);

    }



    u_argv = u_argc + n;

    u_envp = u_argv + (argc + 1) * n;

    u_auxv = u_envp + (envc + 1) * n;

    info->saved_auxv = u_auxv;

    info->arg_start = u_argv;

    info->arg_end = u_argv + argc * n;



    /* This is correct because Linux defines

     * elf_addr_t as Elf32_Off / Elf64_Off

     */

#define NEW_AUX_ENT(id, val) do {               \

        put_user_ual(id, u_auxv);  u_auxv += n; \

        put_user_ual(val, u_auxv); u_auxv += n; \

    } while(0)



    /* There must be exactly DLINFO_ITEMS entries here.  */

#ifdef ARCH_DLINFO

    /*

     * ARCH_DLINFO must come first so platform specific code can enforce

     * special alignment requirements on the AUXV if necessary (eg. PPC).

     */

    ARCH_DLINFO;

#endif

    NEW_AUX_ENT(AT_PHDR, (abi_ulong)(info->load_addr + exec->e_phoff));

    NEW_AUX_ENT(AT_PHENT, (abi_ulong)(sizeof (struct elf_phdr)));

    NEW_AUX_ENT(AT_PHNUM, (abi_ulong)(exec->e_phnum));

    NEW_AUX_ENT(AT_PAGESZ, (abi_ulong)(MAX(TARGET_PAGE_SIZE, getpagesize())));

    NEW_AUX_ENT(AT_BASE, (abi_ulong)(interp_info ? interp_info->load_addr : 0));

    NEW_AUX_ENT(AT_FLAGS, (abi_ulong)0);

    NEW_AUX_ENT(AT_ENTRY, info->entry);

    NEW_AUX_ENT(AT_UID, (abi_ulong) getuid());

    NEW_AUX_ENT(AT_EUID, (abi_ulong) geteuid());

    NEW_AUX_ENT(AT_GID, (abi_ulong) getgid());

    NEW_AUX_ENT(AT_EGID, (abi_ulong) getegid());

    NEW_AUX_ENT(AT_HWCAP, (abi_ulong) ELF_HWCAP);

    NEW_AUX_ENT(AT_CLKTCK, (abi_ulong) sysconf(_SC_CLK_TCK));

    NEW_AUX_ENT(AT_RANDOM, (abi_ulong) u_rand_bytes);



#ifdef ELF_HWCAP2

    NEW_AUX_ENT(AT_HWCAP2, (abi_ulong) ELF_HWCAP2);

#endif



    if (u_platform) {

        NEW_AUX_ENT(AT_PLATFORM, u_platform);

    }

    NEW_AUX_ENT (AT_NULL, 0);

#undef NEW_AUX_ENT



    info->auxv_len = u_argv - info->saved_auxv;



    put_user_ual(argc, u_argc);



    p = info->arg_strings;

    for (i = 0; i < argc; ++i) {

        put_user_ual(p, u_argv);

        u_argv += n;

        p += target_strlen(p) + 1;

    }

    put_user_ual(0, u_argv);



    p = info->env_strings;

    for (i = 0; i < envc; ++i) {

        put_user_ual(p, u_envp);

        u_envp += n;

        p += target_strlen(p) + 1;

    }

    put_user_ual(0, u_envp);



    return sp;

}
