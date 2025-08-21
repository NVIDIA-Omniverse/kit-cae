# Omni CAE Data [omni.cae.data]

This extension introduces the concept of **Data Delegate**.
The Data Delegate API provides an extensible mechanism to add support for handling `CaeFieldArray` prim and its subtypes. Data Delegate
has two sets of APIs: APIs to access raw data referenced by a `CaeFieldArray` prim, and APIs to register delegates that can handle the
*reading* of raw data referenced by a subtype of `CaeFieldArray`. This extension also defines `omni.kit.Command` types called
`Operator Commands`. These commands can be used by algorithms in Kit-CAE for basic data processing
operations needed for supported algorithms. They provide an extensible mechanism that allows extensions to introduce new data models
for handling different types of data in their native representation. In a typical CAE application, one would adopt a
data model to represent data within the system; however, for Kit-CAE, we intentionally avoid making that choice.