def rewrite_query(query):

    mappings = {
        "risks": "risk factors business risks threats affecting Apple operations",
        "headquarters": "company headquarters corporate offices location",
        "products": "main product categories iPhone Mac iPad Services",
        "research and development": "R&D expense research development spending",
        "cybersecurity": "cybersecurity risks malicious attacks unauthorized access information security threats"
    }


    lower_query = query.lower()


    for key, value in mappings.items():

        if key in lower_query:

            return (
                query
                + " "
                + value
            )


    # Keep original query if no rewrite rule applies

    return query