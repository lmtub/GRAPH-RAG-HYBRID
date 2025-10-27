static char *get_content_url(xmlNodePtr *baseurl_nodes,

                             int n_baseurl_nodes,

                             char *rep_id_val,

                             char *rep_bandwidth_val,

                             char *val)

{

    int i;

    char *text;

    char *url = NULL;

    char tmp_str[MAX_URL_SIZE];

    char tmp_str_2[MAX_URL_SIZE];



    memset(tmp_str, 0, sizeof(tmp_str));



    for (i = 0; i < n_baseurl_nodes; ++i) {

        if (baseurl_nodes[i] &&

            baseurl_nodes[i]->children &&

            baseurl_nodes[i]->children->type == XML_TEXT_NODE) {

            text = xmlNodeGetContent(baseurl_nodes[i]->children);

            if (text) {

                memset(tmp_str, 0, sizeof(tmp_str));

                memset(tmp_str_2, 0, sizeof(tmp_str_2));

                ff_make_absolute_url(tmp_str_2, MAX_URL_SIZE, tmp_str, text);

                av_strlcpy(tmp_str, tmp_str_2, sizeof(tmp_str));

                xmlFree(text);

            }

        }

    }



    if (val)

        av_strlcat(tmp_str, (const char*)val, sizeof(tmp_str));



    if (rep_id_val) {

        url = av_strireplace(tmp_str, "$RepresentationID$", (const char*)rep_id_val);

        if (!url) {

            return NULL;

        }

        av_strlcpy(tmp_str, url, sizeof(tmp_str));

        av_free(url);

    }

    if (rep_bandwidth_val && tmp_str[0] != '\0') {

        url = av_strireplace(tmp_str, "$Bandwidth$", (const char*)rep_bandwidth_val);

        if (!url) {

            return NULL;

        }

    }

    return url;

}
