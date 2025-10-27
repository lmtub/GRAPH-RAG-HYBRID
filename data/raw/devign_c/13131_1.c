static inline int target_to_host_errno(int err)

{

    if (target_to_host_errno_table[err])

        return target_to_host_errno_table[err];

    return err;

}
