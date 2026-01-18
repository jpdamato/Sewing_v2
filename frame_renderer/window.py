import moderngl
import glfw
import numpy as np


class Window:
    _shared_ctx: moderngl.Context | None = None
    _shared_glfw_window = None
    _shared_prog = None
    _current_context_window = None

    def __init__(
        self,
        height: int,
        width: int,
        title: str,
        bg_color=[0, 0, 0],
        position=(0, 0),
        maximized=True,
        decorators=False,
        vsync=False,
    ):
        self.width: int = width
        self.height: int = height
        self.title: str = title
        self.x_pos: int = position[0]
        self.y_pos: int = position[1]
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        glfw.window_hint(glfw.VISIBLE, True)
        glfw.window_hint(glfw.RESIZABLE, False)

        if not decorators:
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)

        self.glfw_window = glfw.create_window(
            width, height, title, None, share=Window._shared_glfw_window
        )

        if not self.glfw_window:
            raise RuntimeError(f"Failed to create GLFW window: {title}")

        glfw.make_context_current(self.glfw_window)
        Window._current_context_window = self.glfw_window

        glfw.swap_interval(0 if not vsync else 1)

        if Window._shared_ctx is None:
            Window._shared_glfw_window = self.glfw_window
            Window._shared_ctx = moderngl.create_context()
            Window._shared_prog = Window._shared_ctx.program(
                vertex_shader="""
                    #version 330
                    in vec2 in_vert;
                    out vec2 v_uv;
                    void main() {
                        v_uv = in_vert * 0.5 + 0.5;
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330
                    uniform sampler2D tex;
                    in vec2 v_uv;
                    out vec4 f_color;
                    void main() {
                        f_color = texture(tex, v_uv);
                    }
                """,
            )

        self.ctx = Window._shared_ctx

        self._frame_buffer = np.zeros((height, width, 3), dtype=np.uint8, order="C")
        self._frame_buffer[:] = bg_color

        self.texture = self.ctx.texture(
            (self.width, self.height),
            components=3,
            data=self._frame_buffer.tobytes(),
        )
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.repeat_x = False
        self.texture.repeat_y = False

        self.quad = self.ctx.buffer(
            np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4").tobytes()
        )
        if Window._shared_prog is None:
            raise RuntimeError("Shared program not initialized")

        self.vao = self.ctx.simple_vertex_array(
            Window._shared_prog, self.quad, "in_vert"
        )

        self.set_window_position(position)

        if maximized:
            self.maximize()

    def render(self, frame: np.ndarray):
        if Window._current_context_window is not self.glfw_window:
            glfw.make_context_current(self.glfw_window)
            Window._current_context_window = self.glfw_window

        if frame.flags["C_CONTIGUOUS"]:
            self.texture.write(frame)
        else:
            self.texture.write(np.ascontiguousarray(frame))

        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(self.glfw_window)

    def set_window_position(self, position):
        self.x_pos: int = position[0]
        self.y_pos: int = position[1]
        glfw.set_window_pos(self.glfw_window, position[0], position[1])

    def maximize(self):
        glfw.maximize_window(self.glfw_window)

    def close(self):
        glfw.destroy_window(self.glfw_window)
