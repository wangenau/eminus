..
   SPDX-FileCopyrightText: 2021 The eminus developers
   SPDX-License-Identifier: Apache-2.0

{{ fullname | remove_package_name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
   :members:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
