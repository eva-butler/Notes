# Software Engineering CS3240

## class 3
9/5/2024

GitHub
- main should always be deployable

### Software Process and Methodologies

A process
- find and repeat good practices
- management - > need to know what to do next, and keep things on track

Software Procss:
-  software process is a set of activities that take place in sequence in the pursuit of creation of a software system
  - requirements
  - specifications
  - desgin
  - development
  - validation
  - evolution
- the form/iteration/timing/ect. of these activities defines the different methodoligies
- These methods can fall on the continuum between an Agile and Plan-Driven method

MODELS:
- waterfall
  - do the phases shown before in order
  - complete one phase before moving onto the next
  - a well defined document with the layout
  - very rigid
- spiral
  - basically the waterfall method that can circle back and run through the process multiple times
  - risk analysis and management are explicity shown in the model at each stage
  - repititon of activities in model
  - <img width="214" alt="image" src="https://github.com/user-attachments/assets/fd7cfe56-1865-444e-bcde-8b6303d7c4f0">

- various agile methods
  - Scrum, Extreme Programming, Kanban
- various plan-driven methods
  - Plan-driven methods work best when developers can determine the requirements in advance . . . and when the requirements remain relatively stable, with change rates on the order of one percent per month.”
  - Depending on how they are implemented, they can be more plan-driven or more agile
  - Focus on repeatability and predictability
  - Documentation
  - Rational Unified Process
    - 1980s, built around the UML (unified modeling language), toolsets were created to hep build UML diagrams, and then translate them into code
    - incorportated into multiple IDEs
    - Built around six best practices:
      - Develop software iteratively
      - Manage requirements
      - Use component-based architectures
      - Visually model software
      - Verify software qualit
      - Control changes to software
  - Personal Software Process
  - Team Software Process
  - 
- other families of approaches


# Django Practice:

## part 1)
- django-admin startproject mysite
begin the project
These files are:

The outer mysite/ root directory is a container for your project. Its name doesn’t matter to Django; you can rename it to anything you like.
manage.py: A command-line utility that lets you interact with this Django project in various ways. You can read all the details about manage.py in django-admin and manage.py.
The inner mysite/ directory is the actual Python package for your project. Its name is the Python package name you’ll need to use to import anything inside it (e.g. mysite.urls).
mysite/__init__.py: An empty file that tells Python that this directory should be considered a Python package. If you’re a Python beginner, read more about packages in the official Python docs.
mysite/settings.py: Settings/configuration for this Django project. Django settings will tell you all about how settings work.
mysite/urls.py: The URL declarations for this Django project; a “table of contents” of your Django-powered site. You can read more about URLs in URL dispatcher.
mysite/asgi.py: An entry-point for ASGI-compatible web servers to serve your project. See How to deploy with ASGI for more details.
mysite/wsgi.py: An entry-point for WSGI-compatible web servers to serve your project. See How to deploy with WSGI for more details.


After making a new app:
python manage.py startapp polls
you create the following directory:

polls/
    __init__.py
    admin.py
    apps.py
    migrations/
        __init__.py
    models.py
    tests.py
    views.py
      - the http response is teh mist basic view in Django. You can map the response to a url

make a urls.py to connect to the views.py

The next step is to configure the global URLconf in the mysite project to include the URLconf defined in polls.urls. To do this, add an import for django.urls.include in mysite/urls.py and insert an include() in the urlpatterns list, so you have:

    from django.contrib import admin
    from django.urls import include, path
    
    urlpatterns = [
        path("polls/", include("polls.urls")),
        path("admin/", admin.site.urls),
    ]

You should always use include() when you include other URL patterns. admin.site.urls is the only exception to this.

path()
- The path() function is passed four arguments, two required: route and view, and two optional: kwargs, and name. At this point, it’s worth reviewing what these arguments are for.
  - route: a string that contains a URL pattern. When processing a request, Django starts at the first pattern in urlpatterns and makes its way down the list, comparing the requested URL against each pattern until it finds one that matches.
  - view: When Django finds a matching pattern, it calls the specified view function with an HttpRequest object as the first argument and any “captured” values from the route as keyword arguments.
  - name: Naming your URL lets you refer to it unambiguously from elsewhere in Django, especially from within templates. This powerful feature allows you to make global changes to the URL patterns of your project while only touching a single file.
 
## part 2) Database set up

settings.py
- TIME_ZONE: set it when u edit
- INSTALLED_APPS: names of all Django applications that are activated in this Django instance. Apps can be used in multiple projects, and you can package and distribute them for use by others in their projects. FOlliwing are the default apps included
  - django.contrib.admin – The admin site. You’ll use it shortly.
  - django.contrib.auth – An authentication system.
  - django.contrib.contenttypes – A framework for content types.
  - django.contrib.sessions – A session framework.
  - django.contrib.messages – A messaging framework.
  - django.contrib.staticfiles – A framework for managing static files.
- migrate:
  - The migrate command looks at the INSTALLED_APPS setting and creates any necessary database tables according to the database settings in your mysite/settings.py file and the database migrations shipped with the app (we’ll cover those later). 
Creating Models:
- A model is the single, definitive source of information about your data. It contains the essential fields and behaviors of the data you’re storing. Django follows the DRY Principle. The goal is to define your data model in one place and automatically derive things from it.
- This includes the migrations - unlike in Ruby On Rails, for example, migrations are entirely derived from your models file, and are essentially a history that Django can roll through to update your database schema to match your current models.
- In our poll app, we’ll create two models: Question and Choice. A Question has a question and a publication date. A Choice has two fields: the text of the choice and a vote tally. Each Choice is associated with a Question.
    
      class Question(models.Model):
          question_text = models.CharField(max_length=200)
          pub_date = models.DateTimeField("date published")
      
      
      class Choice(models.Model):
          question = models.ForeignKey(Question, on_delete=models.CASCADE)
          choice_text = models.CharField(max_length=200)
          votes = models.IntegerField(default=0)

- Here, each model is represented by a class that subclasses django.db.models.Model. Each model has a number of class variables, each of which represents a database field in the model.
- Each field is represented by an instance of a Field class – e.g., CharField for character fields and DateTimeField for datetimes. This tells Django what type of data each field holds.
- The name of each Field instance (e.g. question_text or pub_date) is the field’s name, in machine-friendly format. You’ll use this value in your Python code, and your database will use it as the column name.
- Finally, note a relationship is defined, using ForeignKey. That tells Django each Choice is related to a single Question. Django supports all the common database relationships: many-to-one, many-to-many, and one-to-one.

Activating Models:
- Django apps are “pluggable”: You can use an app in multiple projects, and you can distribute apps, because they don’t have to be tied to a given Django installation.
- onfiguration class in the INSTALLED_APPS setting. The PollsConfig class is in the polls/apps.py file, so its dotted path is 'polls.apps.PollsConfig'. Edit the mysite/settings.py file and add that dotted path to the INSTALLED_APPS setting. It’ll look like this:
- makemigrations
  - $ python manage.py makemigrations polls
  - By running makemigrations, you’re telling Django that you’ve made some changes to your models (in this case, you’ve made new ones) and that you’d like the changes to be stored as a migration.
- sqlmigrate:
  - command takes migration names and returns their SQL:
  - python manage.py sqlmigrate polls 0001
  - Table names are automatically generated by combining the name of the app (polls) and the lowercase name of the model – question and choice.
  - Primary keys (IDs) are added automatically. (You can override this, too.)
  - The foreign key relationship is made explicit by a FOREIGN KEY constraint. Don’t worry about the DEFERRABLE parts; it’s telling PostgreSQL to not enforce the foreign key until the end of the transaction.
  - The sqlmigrate command doesn’t actually run the migration on your database - instead, it prints it to the screen so that you can see what SQL Django thinks is required. It’s useful for checking what Django is going to do or if you have database administrators who require SQL scripts for changes.
  - We’ll cover them in more depth in a later part of the tutorial, but for now, remember the three-step guide to making model changes:
    - Change your models (in models.py)
    - Run python manage.py makemigrations to create migrations for those changes
    - Run python manage.py migrate to apply those changes to the database.

Play with the API

    In [1]: from polls.models import Choice, Question
    
    In [2]: Question.objects.all()
    Out[2]: <QuerySet []>
    
    In [3]: from django.utils import timezone
    
    In [4]: q = Question(question_text = "What's new?", pub_date = time
       ...: zone.now())
    
    In [5]: q.save()
    
    In [6]: q.id
    Out[6]: 1
    
    In [7]: q.question_text
    Out[7]: "What's new?"
    
    In [8]: q.save()
    
    In [9]: Questions.objects.all()
    -------------------------------------------------------------------
    NameError                         Traceback (most recent call last)
    Cell In[9], line 1
    ----> 1 Questions.objects.all()
    
    NameError: name 'Questions' is not defined
    
    In [10]: Question.objects.all()
    Out[10]: <QuerySet [<Question: Question object (1)>]>
    
    In [11]: exit
    (base) root@Eva:~/code/CS3240/django-eva-butler# python manage.py shell
    Python 3.11.7 (main, Dec 15 2023, 18:12:31) [GCC 11.2.0]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 8.20.0 -- An enhanced Interactive Python. Type '?' for help.
    
    In [1]: from po
      Cell In[1], line 1
        from po
               ^
    SyntaxError: invalid syntax
    
    
    In [2]: from polls.models import Choice, Question
    
    In [3]: Question
    Out[3]: polls.models.Question
    
    In [4]: Question.objects.all()
    Out[4]: <QuerySet [<Question: What's new?>]>
    
    In [5]: Question.objects.filter(id=1)
    Out[5]: <QuerySet [<Question: What's new?>]>
    
    In [6]: from django.utils import timezone
    
    In [7]: current_year = timezone.now().uear
    -------------------------------------------------------------------
    AttributeError                    Traceback (most recent call last)
    Cell In[7], line 1
    ----> 1 current_year = timezone.now().uear
    
    AttributeError: 'datetime.datetime' object has no attribute 'uear'
    
    In [8]: current_year = timezone.now().year
    
    In [9]: Question.objects.get(pub_date__year=current_year)
    Out[9]: <Question: What's new?>
    
    In [10]: q.choice_set.create(choice_text = "Not much", votes = 0)
    -------------------------------------------------------------------
    NameError                         Traceback (most recent call last)
    Cell In[10], line 1
    ----> 1 q.choice_set.create(choice_text = "Not much", votes = 0)
    
    NameError: name 'q' is not defined
    
    In [11]: q = Question.objects.get(pk=1)
    
    In [12]: q.choice_set.create(choice_text = "Not much", votes = 0)
    Out[12]: <Choice: Not much>
    
    In [13]: q.choice_set.create(choice_text = "The sky", votes = 0)
    Out[13]: <Choice: The sky>
    
    In [14]: c = q.choice_set.create(choice_text = "just hacking again"
        ...: , votes = 0)
    
    In [15]: c.question
    Out[15]: <Question: What's new?>
    
    In [16]: c = q.choice_set.filter(choice_text__startswith="just hack
        ...: ing")
    
    In [17]: c.delete()
    Out[17]: (1, {'polls.Choice': 1})

Introducing the Django Admin
- Generating admin sites for your staff or clients to add, change, and delete content is tedious work that doesn’t require much creativity. For that reason, Django entirely automates creation of admin interfaces for models.

      (base) root@Eva:~/code/CS3240/django-eva-butler# python manage.py createsuperuser
      Username (leave blank to use 'root'): admin
      Email address: admin@example.com
      Password: 
      Password (again): 
      Superuser created successfully.

      from django.contrib import admin
      
      from .models import Question
      
      admin.site.register(Question)

  - this makes the poll modifyable in the admin


  ## part 3)
  
