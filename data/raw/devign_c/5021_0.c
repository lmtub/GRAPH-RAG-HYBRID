static void vtd_root_table_setup(IntelIOMMUState *s)

{

    s->root = vtd_get_quad_raw(s, DMAR_RTADDR_REG);

    s->root_extended = s->root & VTD_RTADDR_RTT;

    s->root &= VTD_RTADDR_ADDR_MASK(VTD_HOST_ADDRESS_WIDTH);



    trace_vtd_reg_dmar_root(s->root, s->root_extended);

}
