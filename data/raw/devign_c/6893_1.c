void gdb_register_coprocessor(CPUState * env,

                             gdb_reg_cb get_reg, gdb_reg_cb set_reg,

                             int num_regs, const char *xml, int g_pos)

{

    GDBRegisterState *s;

    GDBRegisterState **p;

    static int last_reg = NUM_CORE_REGS;



    s = (GDBRegisterState *)g_malloc0(sizeof(GDBRegisterState));

    s->base_reg = last_reg;

    s->num_regs = num_regs;

    s->get_reg = get_reg;

    s->set_reg = set_reg;

    s->xml = xml;

    p = &env->gdb_regs;

    while (*p) {

        /* Check for duplicates.  */

        if (strcmp((*p)->xml, xml) == 0)

            return;

        p = &(*p)->next;

    }

    /* Add to end of list.  */

    last_reg += num_regs;

    *p = s;

    if (g_pos) {

        if (g_pos != s->base_reg) {

            fprintf(stderr, "Error: Bad gdb register numbering for '%s'\n"

                    "Expected %d got %d\n", xml, g_pos, s->base_reg);

        } else {

            num_g_regs = last_reg;

        }

    }

}
