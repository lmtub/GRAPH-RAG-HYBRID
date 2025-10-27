S390CPU *cpu_s390x_init(const char *cpu_model)

{

    S390CPU *cpu;



    cpu = S390_CPU(object_new(TYPE_S390_CPU));



    object_property_set_bool(OBJECT(cpu), true, "realized", NULL);



    return cpu;

}
