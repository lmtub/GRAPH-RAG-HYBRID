static abi_long do_ioctl_rt(const IOCTLEntry *ie, uint8_t *buf_temp,

                                int fd, int cmd, abi_long arg)

{

    const argtype *arg_type = ie->arg_type;

    const StructEntry *se;

    const argtype *field_types;

    const int *dst_offsets, *src_offsets;

    int target_size;

    void *argptr;

    abi_ulong *target_rt_dev_ptr;

    unsigned long *host_rt_dev_ptr;

    abi_long ret;

    int i;



    assert(ie->access == IOC_W);

    assert(*arg_type == TYPE_PTR);

    arg_type++;

    assert(*arg_type == TYPE_STRUCT);

    target_size = thunk_type_size(arg_type, 0);

    argptr = lock_user(VERIFY_READ, arg, target_size, 1);

    if (!argptr) {

        return -TARGET_EFAULT;

    }

    arg_type++;

    assert(*arg_type == (int)STRUCT_rtentry);

    se = struct_entries + *arg_type++;

    assert(se->convert[0] == NULL);

    /* convert struct here to be able to catch rt_dev string */

    field_types = se->field_types;

    dst_offsets = se->field_offsets[THUNK_HOST];

    src_offsets = se->field_offsets[THUNK_TARGET];

    for (i = 0; i < se->nb_fields; i++) {

        if (dst_offsets[i] == offsetof(struct rtentry, rt_dev)) {

            assert(*field_types == TYPE_PTRVOID);

            target_rt_dev_ptr = (abi_ulong *)(argptr + src_offsets[i]);

            host_rt_dev_ptr = (unsigned long *)(buf_temp + dst_offsets[i]);

            if (*target_rt_dev_ptr != 0) {

                *host_rt_dev_ptr = (unsigned long)lock_user_string(

                                                  tswapal(*target_rt_dev_ptr));

                if (!*host_rt_dev_ptr) {

                    unlock_user(argptr, arg, 0);

                    return -TARGET_EFAULT;

                }

            } else {

                *host_rt_dev_ptr = 0;

            }

            field_types++;

            continue;

        }

        field_types = thunk_convert(buf_temp + dst_offsets[i],

                                    argptr + src_offsets[i],

                                    field_types, THUNK_HOST);

    }

    unlock_user(argptr, arg, 0);

    assert(host_rt_dev_ptr);



    ret = get_errno(safe_ioctl(fd, ie->host_cmd, buf_temp));

    if (*host_rt_dev_ptr != 0) {

        unlock_user((void *)*host_rt_dev_ptr,

                    *target_rt_dev_ptr, 0);

    }

    return ret;

}
