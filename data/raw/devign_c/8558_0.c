static bool pmsav7_rgnr_vmstate_validate(void *opaque, int version_id)

{

    ARMCPU *cpu = opaque;



    return cpu->env.pmsav7.rnr < cpu->pmsav7_dregion;

}
