static abi_long do_ioctl_dm(const IOCTLEntry *ie, uint8_t *buf_temp, int fd,

                            abi_long cmd, abi_long arg)

{

    void *argptr;

    struct dm_ioctl *host_dm;

    abi_long guest_data;

    uint32_t guest_data_size;

    int target_size;

    const argtype *arg_type = ie->arg_type;

    abi_long ret;

    void *big_buf = NULL;

    char *host_data;



    arg_type++;

    target_size = thunk_type_size(arg_type, 0);

    argptr = lock_user(VERIFY_READ, arg, target_size, 1);

    if (!argptr) {

        ret = -TARGET_EFAULT;

        goto out;

    }

    thunk_convert(buf_temp, argptr, arg_type, THUNK_HOST);

    unlock_user(argptr, arg, 0);



    /* buf_temp is too small, so fetch things into a bigger buffer */

    big_buf = g_malloc0(((struct dm_ioctl*)buf_temp)->data_size * 2);

    memcpy(big_buf, buf_temp, target_size);

    buf_temp = big_buf;

    host_dm = big_buf;



    guest_data = arg + host_dm->data_start;

    if ((guest_data - arg) < 0) {

        ret = -EINVAL;

        goto out;

    }

    guest_data_size = host_dm->data_size - host_dm->data_start;

    host_data = (char*)host_dm + host_dm->data_start;



    argptr = lock_user(VERIFY_READ, guest_data, guest_data_size, 1);

    switch (ie->host_cmd) {

    case DM_REMOVE_ALL:

    case DM_LIST_DEVICES:

    case DM_DEV_CREATE:

    case DM_DEV_REMOVE:

    case DM_DEV_SUSPEND:

    case DM_DEV_STATUS:

    case DM_DEV_WAIT:

    case DM_TABLE_STATUS:

    case DM_TABLE_CLEAR:

    case DM_TABLE_DEPS:

    case DM_LIST_VERSIONS:

        /* no input data */

        break;

    case DM_DEV_RENAME:

    case DM_DEV_SET_GEOMETRY:

        /* data contains only strings */

        memcpy(host_data, argptr, guest_data_size);

        break;

    case DM_TARGET_MSG:

        memcpy(host_data, argptr, guest_data_size);

        *(uint64_t*)host_data = tswap64(*(uint64_t*)argptr);

        break;

    case DM_TABLE_LOAD:

    {

        void *gspec = argptr;

        void *cur_data = host_data;

        const argtype arg_type[] = { MK_STRUCT(STRUCT_dm_target_spec) };

        int spec_size = thunk_type_size(arg_type, 0);

        int i;



        for (i = 0; i < host_dm->target_count; i++) {

            struct dm_target_spec *spec = cur_data;

            uint32_t next;

            int slen;



            thunk_convert(spec, gspec, arg_type, THUNK_HOST);

            slen = strlen((char*)gspec + spec_size) + 1;

            next = spec->next;

            spec->next = sizeof(*spec) + slen;

            strcpy((char*)&spec[1], gspec + spec_size);

            gspec += next;

            cur_data += spec->next;

        }

        break;

    }

    default:

        ret = -TARGET_EINVAL;

        goto out;

    }

    unlock_user(argptr, guest_data, 0);



    ret = get_errno(ioctl(fd, ie->host_cmd, buf_temp));

    if (!is_error(ret)) {

        guest_data = arg + host_dm->data_start;

        guest_data_size = host_dm->data_size - host_dm->data_start;

        argptr = lock_user(VERIFY_WRITE, guest_data, guest_data_size, 0);

        switch (ie->host_cmd) {

        case DM_REMOVE_ALL:

        case DM_DEV_CREATE:

        case DM_DEV_REMOVE:

        case DM_DEV_RENAME:

        case DM_DEV_SUSPEND:

        case DM_DEV_STATUS:

        case DM_TABLE_LOAD:

        case DM_TABLE_CLEAR:

        case DM_TARGET_MSG:

        case DM_DEV_SET_GEOMETRY:

            /* no return data */

            break;

        case DM_LIST_DEVICES:

        {

            struct dm_name_list *nl = (void*)host_dm + host_dm->data_start;

            uint32_t remaining_data = guest_data_size;

            void *cur_data = argptr;

            const argtype arg_type[] = { MK_STRUCT(STRUCT_dm_name_list) };

            int nl_size = 12; /* can't use thunk_size due to alignment */



            while (1) {

                uint32_t next = nl->next;

                if (next) {

                    nl->next = nl_size + (strlen(nl->name) + 1);

                }

                if (remaining_data < nl->next) {

                    host_dm->flags |= DM_BUFFER_FULL_FLAG;

                    break;

                }

                thunk_convert(cur_data, nl, arg_type, THUNK_TARGET);

                strcpy(cur_data + nl_size, nl->name);

                cur_data += nl->next;

                remaining_data -= nl->next;

                if (!next) {

                    break;

                }

                nl = (void*)nl + next;

            }

            break;

        }

        case DM_DEV_WAIT:

        case DM_TABLE_STATUS:

        {

            struct dm_target_spec *spec = (void*)host_dm + host_dm->data_start;

            void *cur_data = argptr;

            const argtype arg_type[] = { MK_STRUCT(STRUCT_dm_target_spec) };

            int spec_size = thunk_type_size(arg_type, 0);

            int i;



            for (i = 0; i < host_dm->target_count; i++) {

                uint32_t next = spec->next;

                int slen = strlen((char*)&spec[1]) + 1;

                spec->next = (cur_data - argptr) + spec_size + slen;

                if (guest_data_size < spec->next) {

                    host_dm->flags |= DM_BUFFER_FULL_FLAG;

                    break;

                }

                thunk_convert(cur_data, spec, arg_type, THUNK_TARGET);

                strcpy(cur_data + spec_size, (char*)&spec[1]);

                cur_data = argptr + spec->next;

                spec = (void*)host_dm + host_dm->data_start + next;

            }

            break;

        }

        case DM_TABLE_DEPS:

        {

            void *hdata = (void*)host_dm + host_dm->data_start;

            int count = *(uint32_t*)hdata;

            uint64_t *hdev = hdata + 8;

            uint64_t *gdev = argptr + 8;

            int i;



            *(uint32_t*)argptr = tswap32(count);

            for (i = 0; i < count; i++) {

                *gdev = tswap64(*hdev);

                gdev++;

                hdev++;

            }

            break;

        }

        case DM_LIST_VERSIONS:

        {

            struct dm_target_versions *vers = (void*)host_dm + host_dm->data_start;

            uint32_t remaining_data = guest_data_size;

            void *cur_data = argptr;

            const argtype arg_type[] = { MK_STRUCT(STRUCT_dm_target_versions) };

            int vers_size = thunk_type_size(arg_type, 0);



            while (1) {

                uint32_t next = vers->next;

                if (next) {

                    vers->next = vers_size + (strlen(vers->name) + 1);

                }

                if (remaining_data < vers->next) {

                    host_dm->flags |= DM_BUFFER_FULL_FLAG;

                    break;

                }

                thunk_convert(cur_data, vers, arg_type, THUNK_TARGET);

                strcpy(cur_data + vers_size, vers->name);

                cur_data += vers->next;

                remaining_data -= vers->next;

                if (!next) {

                    break;

                }

                vers = (void*)vers + next;

            }

            break;

        }

        default:

            ret = -TARGET_EINVAL;

            goto out;

        }

        unlock_user(argptr, guest_data, guest_data_size);



        argptr = lock_user(VERIFY_WRITE, arg, target_size, 0);

        if (!argptr) {

            ret = -TARGET_EFAULT;

            goto out;

        }

        thunk_convert(argptr, buf_temp, arg_type, THUNK_TARGET);

        unlock_user(argptr, arg, target_size);

    }

out:

    if (big_buf) {

        free(big_buf);

    }

    return ret;

}
