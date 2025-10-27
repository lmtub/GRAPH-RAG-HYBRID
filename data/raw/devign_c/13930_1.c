void syscall_init(void)
{
    IOCTLEntry *ie;
    const argtype *arg_type;
    int size;
    int i;
#define STRUCT(name, ...) thunk_register_struct(STRUCT_ ## name, #name, struct_ ## name ## _def);
#define STRUCT_SPECIAL(name) thunk_register_struct_direct(STRUCT_ ## name, #name, &struct_ ## name ## _def);
#include "syscall_types.h"
#undef STRUCT
#undef STRUCT_SPECIAL
    /* Build target_to_host_errno_table[] table from
     * host_to_target_errno_table[]. */
    for (i = 0; i < ERRNO_TABLE_SIZE; i++) {
        target_to_host_errno_table[host_to_target_errno_table[i]] = i;
    }
    /* we patch the ioctl size if necessary. We rely on the fact that
       no ioctl has all the bits at '1' in the size field */
    ie = ioctl_entries;
    while (ie->target_cmd != 0) {
        if (((ie->target_cmd >> TARGET_IOC_SIZESHIFT) & TARGET_IOC_SIZEMASK) ==
            TARGET_IOC_SIZEMASK) {
            arg_type = ie->arg_type;
            if (arg_type[0] != TYPE_PTR) {
                fprintf(stderr, "cannot patch size for ioctl 0x%x\n",
                        ie->target_cmd);
                exit(1);
            }
            arg_type++;
            size = thunk_type_size(arg_type, 0);
            ie->target_cmd = (ie->target_cmd &
                              ~(TARGET_IOC_SIZEMASK << TARGET_IOC_SIZESHIFT)) |
                (size << TARGET_IOC_SIZESHIFT);
        }
        /* automatic consistency check if same arch */
#if (defined(__i386__) && defined(TARGET_I386) && defined(TARGET_ABI32)) || \
    (defined(__x86_64__) && defined(TARGET_X86_64))
        if (unlikely(ie->target_cmd != ie->host_cmd)) {
            fprintf(stderr, "ERROR: ioctl(%s): target=0x%x host=0x%x\n",
                    ie->name, ie->target_cmd, ie->host_cmd);
        }
#endif
        ie++;
    }
}