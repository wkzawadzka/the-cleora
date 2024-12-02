import html
class DataPipeline():
    def __init__(self, steps):
        self.steps = steps  # List of (name, step) tuples, where each step is a callable

    def run(self, data=None):
        for name, step in self.steps:
            print(f"Running step: {name}")
            data = step(data)  # Execute the step and update the data
        return data
    
    def to_html(self):
        """
        Generate an HTML representation of the pipeline.
        Returns:
            A string containing HTML code that represents the pipeline.
        """
        html_representation = """
        <div style='font-family: Arial;'>
            <h2>DataPipeline</h2>
            <ol>
        """
        for name, step in self.steps:
            step_name = html.escape(str(type(step).__name__))
            step_details = html.escape(str(step))
            html_representation += f"""
                <li>
                    <strong>{html.escape(name)}</strong>: 
                    <span style="color: #555;">{step_name}</span>
                    <br/>
                    <code style="background: #f9f9f9; padding: 2px 4px; border-radius: 4px;">
                        {step_details}
                    </code>
                </li>
            """
        html_representation += """
            </ol>
        </div>
        """
        return html_representation
    
    def to_html2(self):
        """
        Generate an HTML representation of the pipeline (based on sklearn).
        Returns:
            HTML string.
        """
        unique_id = f"sk-container-id-222" 
        html_content = f"""
        <style>
        #{unique_id} {{
          font-family: Arial, sans-serif;
          --sklearn-color-background: #f9f9f9;
          --sklearn-color-border: #cccccc;
          --sklearn-color-header: #4f4f4f;
          --sklearn-color-text: #333333;
          border: 1px solid var(--sklearn-color-border);
          border-radius: 5px;
          padding: 10px;
          background: var(--sklearn-color-background);
        }}
        #{unique_id} .step {{
          border: 1px solid var(--sklearn-color-border);
          border-radius: 3px;
          padding: 5px;
          margin: 5px 0;
          background: white;
        }}
        #{unique_id} .step-title {{
          font-weight: bold;
          color: var(--sklearn-color-header);
        }}
        #{unique_id} .step-details {{
          margin-left: 10px;
          font-family: monospace;
          color: var(--sklearn-color-text);
        }}
        </style>
        <div id="{unique_id}">
            <h3>DataPipeline</h3>
            <ol>
        """
        for name, step in self.steps:
            step_type = type(step).__name__
            step_repr = str(step)
            html_content += f"""
                <li class="step">
                    <div class="step-title">{name} ({step_type})</div>
                    <div class="step-details">{step_repr}</div>
                </li>
            """
        html_content += """
            </ol>
        </div>
        """
        return html_content