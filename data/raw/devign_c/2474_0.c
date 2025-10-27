static sPAPRCapabilities default_caps_with_cpu(sPAPRMachineState *spapr,

                                               CPUState *cs)

{

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(spapr);

    sPAPRCapabilities caps;



    caps = smc->default_caps;



    /* TODO: clamp according to cpu model */



    return caps;

}
