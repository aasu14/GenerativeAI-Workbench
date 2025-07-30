# Create a search index in the Azure portal

## Prerequisites

An active Azure subscription.

An existing Azure AI Search service (formerly known as Azure Cognitive Search).

### Step 1: Navigate to Your Search Service

Sign in to the Azure portal.

Locate your Azure AI Search service. You can find it by searching for "AI Search" in the search bar or by navigating to it from your list of resources.

On the search service's Overview page, you'll see a summary of your service, including indexes, indexers, and data sources.

### Step 2: Add a New Index

In the left-hand navigation pane of your search service page, under the Search management section, select Indexes.

Click the + Add Index button at the top of the Indexes pane. This will open the new index configuration page.

### Step 3: Define the Index Schema (Fields)

This is where you define the structure of your search documents. The index is composed of a collection of fields.

Index Name: Assign a unique, lowercase name for your index. The name must start with a letter and can contain letters, numbers, and dashes.

Fields: Click + Add Field for each field you want in your index. For each field, you must specify:

Name: The name of the field (e.g., hotelId, hotelName, description, rating).

Data Type: The type of data the field will hold (e.g., Edm.String, Edm.Int32, Edm.Boolean, Edm.GeographyPoint, Collection(Edm.String)).

Attributes: Check the boxes for the behaviors you want to enable for each field.

🔑 Key: Required. Exactly one field (of type Edm.String) must be designated as the key. This field uniquely identifies each document in the index.

Retrievable: The field's value can be returned in search results.

Filterable: Allows for filtering on this field (e.g., rating gt 4).

Sortable: Allows search results to be sorted based on this field.

Facetable: Allows the field to be used in faceted navigation, which helps users refine search results.

Searchable: The field is included in full-text search queries. Only applies to string and collection types. You can also select a language analyzer for the field from the Analyzer dropdown.

Example Field Configuration:
| Field Name | Data Type | Key | Retrievable | Filterable | Sortable | Facetable | Searchable |
| :--- | :--- | :-: | :---: | :---: | :---: | :---: | :---: |
| hotelId | Edm.String | ✔️ | ✔️ | | | | |
| hotelName | Edm.String | | ✔️ | | ✔️ | | ✔️ |
| description | Edm.String | | ✔️ | | | | ✔️ |
| rating | Edm.Int32 | | ✔️ | ✔️ | ✔️ | ✔️ | |
| tags | Collection(Edm.String) | | ✔️ | ✔️ | | ✔️ | ✔️ |

### Step 4: Configure Optional Features

While on the new index creation page, you can also configure advanced features.

Suggesters: Create suggesters to enable autocomplete or query suggestion functionality. You need to specify a name for the suggester and select the source fields that will provide the suggested terms.

Scoring Profiles: Define custom scoring profiles if you want to influence the relevance of search results based on specific criteria (e.g., boost documents with a higher rating).

CORS (Cross-Origin Resource Sharing): If your search index will be queried by a web browser-based application from a different domain, enable CORS by selecting Allow all origins or specifying allowed domains.

### Step 5: Create the Index

After defining all your fields and optional configurations, double-check your settings.

Click the Create button at the bottom of the page.
