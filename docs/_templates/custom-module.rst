{{ fullname | remove_package_name | escape | underline}}

.. automodule:: {{ fullname }}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :nosignatures:
      :template: custom-class.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :nosignatures:
      :template: custom-function.rst
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Data') }}

   {% for item in attributes %}
   .. autodata:: {{ fullname }}.{{ item }}
      :annotation:

   {%- endfor %}
   {% endif %}
   {% endblock %}
