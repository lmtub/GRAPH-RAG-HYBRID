bool hpet_find(void)

{

    return object_resolve_path_type("", TYPE_HPET, NULL);

}
