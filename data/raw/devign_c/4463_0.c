static inline bool object_property_is_link(ObjectProperty *prop)

{

    return strstart(prop->type, "link<", NULL);

}
