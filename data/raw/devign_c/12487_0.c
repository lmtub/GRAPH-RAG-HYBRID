static int find_hw_breakpoint(target_ulong addr, int len, int type)

{

    int n;



    for (n = 0; n < nb_hw_breakpoint; n++)

        if (hw_breakpoint[n].addr == addr && hw_breakpoint[n].type == type &&

            (hw_breakpoint[n].len == len || len == -1))

            return n;

    return -1;

}
