static void s390x_cpu_get_id(Object *obj, Visitor *v, const char *name,

                             void *opaque, Error **errp)

{

    S390CPU *cpu = S390_CPU(obj);

    int64_t value = cpu->id;



    visit_type_int(v, name, &value, errp);

}
