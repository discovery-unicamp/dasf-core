.. _principles:

==========================
Principles
==========================

The growth in the use of machine learning techniques has led to the emergence of a significant number of frameworks, libraries and tools in recent times. Depending on the technique used or the purpose of the project, there will possibly be a way to develop something using what already exists. With the further growth of deep learning techniques, more of these facilities become available.

One of the problems with these deep learning techniques is the use of data in batch format. So a large piece of data is subdivided into smaller pieces and iterated during epoch training. Today, there are no tools that process data distributedly on demand in full machine learning pipelines. There are also no tools that still use the maximum computational power using GPUs, for example.

Taking advantage of this niche space to be explored, the DASF was created whose recursive acronym is DASF is an Accelerated and Scalable Framework. The project seeks to fill this gap in creating machine learning pipelines using large volumes of data without dividing them into batches.

So that this was also possible, a series of libraries were gathered that could compose the framework, composing most of the functionalities proposed by it. Such tools will be specified in the next sections.

DASF as a Simple API
----------------------

DASF tries to enable a simple API for the user to use. The idea is to make the user's life easier when using the framework. We believe that the user should not have to worry about the details of the framework, but rather focus on the problem to be solved. The framework should be transparent to the user.

In order to simplify the learning-curve some concepts were created to facilitate the use of the framework. The main ones are:

* **Standard API**: We try to follow the same API as scikit-learn, so that the user does not have to learn a new API to use the framework. This is a very popular API and is widely used in the community. So, operations in DASF usually implement the same methods as scikit-learn, such as fit, predict, transform, etc.
* **Extensibility to new devices**: DASF allows simple extensibility to be used in new devices (e.g., GPU) by implementing a simple interface. This allows the user to use the framework in different devices without having to worry about the details of the implementation.
* **Extensibility to scale**: For multi-node scalability we use the DASK construct graphs under the hood. This allows the user to use the framework in a distributed way without having to worry about the details of the implementation. 