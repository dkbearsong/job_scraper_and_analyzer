import asyncio
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Button
from textual.containers import Vertical

class TestApp(App):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Button("Populate", id="pop"),
            DataTable(id="tbl")
        )
    def on_button_pressed(self, event):
        if event.button.id == "pop":
            t = self.query_one("#tbl", DataTable)
            if not t.columns:
                t.add_columns("Col1", "Col2")
            t.clear()
            t.add_row("Value1", "Value2", key="1")
            t.add_row("Value3", "Value4", key="2")
            print("Row count is:", len(t.rows))
            self.exit()

if __name__ == '__main__':
    app = TestApp()
    asyncio.run(app.run_async())
