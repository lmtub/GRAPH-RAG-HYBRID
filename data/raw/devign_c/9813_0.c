bool gs_allowed(void)

{

    if (kvm_enabled()) {

        MachineClass *mc = MACHINE_GET_CLASS(qdev_get_machine());

        if (object_class_dynamic_cast(OBJECT_CLASS(mc),

                                      TYPE_S390_CCW_MACHINE)) {

            S390CcwMachineClass *s390mc = S390_MACHINE_CLASS(mc);



            return s390mc->gs_allowed;

        }

        /* Make sure the "none" machine can have gs */

        return true;

    }

    return false;

}
